from collections import OrderedDict
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from itertools import zip_longest
from typing import List, Tuple, Type, Union

from .utils import (_is_list, _is_optional, _issubclass_safe, _is_union)

class TensorLayout:
    def read(self, adapter, pos, tensor, argmax=None):
        raise NotImplementedError()

    def write(self, adapter, pos, tensor, val):
        raise NotImplementedError()

@dataclass
class ChunkPrimitive(TensorLayout):
    elem: Union[int, float, bool]

    def __len__(self):
        return 1

    def write(self, _adapter, pos, tensor, val):
        tensor[pos] = val

    def read(self, adapter, pos, tensor, argmax=None):
        return self.elem(adapter.get(tensor, pos))

@dataclass
class ChunkEnum(TensorLayout):
    elem: Enum

    def __len__(self):
        return len(self.elem)

    def write(self, _adapter, pos, tensor, val):
        for i, option in enumerate(list(self.elem)):
            if val == option:
                # here we rely on 1. being automatically converted to 1 
                # when working with int or long tensor dtype
                tensor[pos+i] = 1.
                return
        raise ValueError(f"{val} is not a valid option for {self.elem} enum") 

    def read(self, adapter, pos, tensor, argmax=None):
        options = list(self.elem)
        # might be already computed previously (e.g. in case of Optional)
        if argmax is not None: return options[argmax]
        return options[adapter.argmax(tensor[pos:pos+len(self)])]

@dataclass
class ChunkOptional(TensorLayout):
    elem: Type[TensorLayout]

    def __len__(self):
        return 1 + len(self.elem)

    def write(self, adapter, pos, tensor, val):
        if val is None:
            tensor[pos] = 1.
        else:
            self.elem.write(adapter, pos+1, tensor, val)

    def read(self, adapter, pos, tensor, argmax=None):
        argmax = adapter.argmax(tensor[pos:pos+len(self)])
        if argmax == 0: return None
        return self.elem.read(adapter, pos+1, tensor, argmax=argmax-1)


@dataclass
class ChunkCollection(TensorLayout):
    num: int
    elem: Type[TensorLayout]

    def __len__(self):
        return self.num * len(self.elem)

    def write(self, adapter, pos, tensor, val):
        elem_size = len(self.elem)
        for i, elem_val in zip_longest(range(self.num), val):
            self.elem.write(adapter, pos + i*elem_size, tensor, elem_val)

    def read(self, adapter, pos, tensor, argmax=None):
        # prepare array to avoid re-allocations 
        vals = [None]*self.num
        elem_size = len(self.elem)
        for i in range(self.num):
            vals[i] = self.elem.read(adapter, pos+i*elem_size, tensor)
        return vals

@dataclass
class ChunkUnion(TensorLayout):
    elems: List[Tuple[type, Type[TensorLayout]]] = field(default_factory=list)
    cursor: int = 0
    positions: List[int] = field(default_factory=list)
    num_options: int = 0

    def add(self, cls: type, layout: Type[TensorLayout]):
        self.positions.append(self.cursor)
        self.elems.append((cls, layout))
        self.cursor += len(layout)
        self.num_options += 1

    def __len__(self):
        return self.cursor + self.num_options

    def write(self, adapter, pos, tensor, val):
        for option, (cls, elem) in enumerate(self.elems):
            if isinstance(val, cls):
                tensor[pos+option] = 1.
                elem_pos = self.positions[option]
                elem.write(adapter, self.num_options+elem_pos, tensor, val)
                return
        raise ValueError(f"{type(val)} is not compatible with Union arguments")

    def read(self, adapter, pos, tensor, argmax=None):
        option = adapter.argmax(tensor[pos:pos+self.num_options])
        elem_pos = self.positions[option]
        _, elem = self.elems[option]
        return elem.read(adapter, self.num_options+elem_pos, tensor) 

@dataclass
class ChunkDataclass(TensorLayout):
    cls: type
    cursor: int = 0
    elems: OrderedDict = field(default_factory=OrderedDict)
    positions: List[int] = field(default_factory=list)

    def add(self, name: str, layout: Type[TensorLayout]):
        self.positions.append(self.cursor)
        self.elems[name] = layout
        self.cursor += len(layout)

    def __len__(self):
        return self.cursor
    
    def write(self, adapter, pos, tensor, val):
        for ((k, elem), elem_pos) in zip(self.elems.items(), self.positions):
            elem.write(adapter, pos+elem_pos, tensor, getattr(val, k))

    def read(self, adapter, pos, tensor, argmax=None):
        kvs = {}
        for ((k, elem), elem_pos) in zip(self.elems.items(), self.positions):
            kvs[k] = elem.read(adapter, pos+elem_pos, tensor)
        return self.cls(**kvs)

def _type_layout(type_, metadata=None):
    print(type_)
    if type_ in (int, float, bool):
        return ChunkPrimitive(type_)

    if _issubclass_safe(type_, Enum):
        return ChunkEnum(type_)
    
    if is_dataclass(type_):
        return _dataclass_layout(type_)
    
    if _is_optional(type_) and len(type_.__args__) == 2:
        return ChunkOptional(_type_layout(type_.__args__[0]))
    
    if _is_list(type_):
        arg = type_.__args__[0]
        shape = metadata.get("shape", None)
        if shape is None or shape == []:
            raise ValueError("Shape is not specified for a list field")
        if isinstance(shape, int):
            shape = [shape]
        return ChunkCollection(shape[0], _type_layout(arg, {"shape": shape[1:]}))
    
    if _is_union(type_):
        chunk = ChunkUnion()
        for arg in type_.__args__:
            chunk.add(arg, _type_layout(arg))
        return chunk

    raise ValueError(f"{type_} type is not supported")

def _dataclass_layout(cls):
    dataclass_layout = ChunkDataclass(cls)
    for field in fields(cls):
        dataclass_layout.add(field.name, _type_layout(field.type, field.metadata))
    return dataclass_layout 
