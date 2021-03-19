import abc

from typing import Iterable, Optional, Type, Union

from .adapters import (TensorAdapter, _numpy_adapter, _pytorch_adapter, _tf_adapter)
from .layout import (TensorLayout, _dataclass_layout)

class DataClassTensorMixin(abc.ABC):
   
    def to_numpy(self, *, tensor_layout: Optional[Type[TensorLayout]]=None, dtype=None):
        layout = tensor_layout or self.tensor_layout()
        return _to_tensor(_numpy_adapter,  layout, self, self._resolve_dtype(dtype))

    @classmethod
    def from_numpy(cls, tensor, tensor_layout: Optional[Type[TensorLayout]]=None):
        return _from_tensor(_numpy_adapter, tensor_layout or cls.tensor_layout(), tensor)

    def to_torch(self, *, tensor_layout: Optional[Type[TensorLayout]]=None, dtype=None):
        layout = tensor_layout or self.tensor_layout()
        return _to_tensor(_pytorch_adapter, layout, self, self._resolve_dtype(dtype))

    @classmethod
    def from_torch(cls, tensor, tensor_layout: Optional[Type[TensorLayout]]=None):
        return _from_tensor(_pytorch_adapter, tensor_layout or cls.tensor_layout(), tensor)

    def to_tf(self, *, tensor_layout: Optional[Type[TensorLayout]]=None, dtype=None):
        layout = tensor_layout or self.tensor_layout()
        return _to_tensor(_tf_adapter, layout, self, self._resolve_dtype(dtype))

    @classmethod
    def from_tf(cls, tensor, tensor_layout: Optional[Type[TensorLayout]]=None):
        return _from_tensor(_tf_adapter, tensor_layout or cls.tensor_layout(), tensor)

    @classmethod
    def tensor_layout(cls):
        return _dataclass_layout(cls)

    @classmethod
    def _resolve_dtype(cls, dtype):
        return dtype or cls._default_tensor_dtype or "float32"

def dataclass_tensor(_cls=None, *, dtype="float32"):
    """
    Based on the code in the `dataclasses` module to handle optional-parens
    decorators. See example below:

    @dataclass_tensor
    @dataclass_tensor(dtype="int64")
    class Example:
        ...
    """
    def wrap(cls):
        return _process_class(cls, dtype)

    if _cls is None: return wrap
    return wrap(_cls)

def _process_class(cls, dtype):
    cls.to_numpy = DataClassTensorMixin.to_numpy
    cls.from_numpy = classmethod(DataClassTensorMixin.from_numpy.__func__)
    cls.to_torch = DataClassTensorMixin.to_torch
    cls.from_torch = classmethod(DataClassTensorMixin.from_torch.__func__)
    cls.to_tf = DataClassTensorMixin.to_tf
    cls.from_tf = classmethod(DataClassTensorMixin.from_tf.__func__)
    cls.tensor_layout = classmethod(DataClassTensorMixin.tensor_layout.__func__)
    cls._default_tensor_dtype = dtype
    cls._resolve_dtype = classmethod(DataClassTensorMixin._resolve_dtype.__func__)
    DataClassTensorMixin.register(cls)
    return cls

def config(shape: Optional[Iterable[int]]):
    return {"shape": shape}

def _to_tensor(adapter: TensorAdapter, layout: Type[TensorLayout], val, dtype="float"):
    tensor = adapter.zeros(len(layout), dtype=dtype)
    layout.write(adapter, 0, tensor, val)
    return tensor

def _from_tensor(adapter: TensorAdapter, layout: Type[TensorLayout], tensor):
    return layout.read(adapter, 0, tensor)
