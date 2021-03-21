import abc

from typing import Iterable, Optional, Type, Union

from .adapters import (TensorAdapter, _numpy_adapter, _pytorch_adapter)
from .layout import (TensorLayout, _dataclass_layout)
from .utils import hybridmethod

class DataClassTensorMixin(abc.ABC):
    @hybridmethod 
    def to_numpy(cls,
                 self,
                 obj=None,
                 *,
                 tensor_layout: Optional[Type[TensorLayout]] = None,
                 dtype = None,
                 batch: bool = False,
                 batch_size: Optional[int] = None):
        layout = tensor_layout or cls.tensor_layout()
        return _to_tensor(_numpy_adapter,
                          layout,
                          obj or self,
                          dtype=cls._resolve_dtype(dtype),
                          batch=batch,
                          batch_size=batch_size)

    @classmethod
    def from_numpy(cls, 
                   tensor,
                   *,
                   tensor_layout: Optional[Type[TensorLayout]]=None,
                   batch: bool = False,
                   batch_size: Optional[int] = None):
        return _from_tensor(_numpy_adapter,
                            tensor_layout or cls.tensor_layout(),
                            tensor,
                            batch=batch,
                            batch_size=batch_size)

    @hybridmethod
    def to_torch(cls,
                 self,
                 obj=None,
                 *,
                 tensor_layout: Optional[Type[TensorLayout]] = None,
                 dtype = None,
                 batch: bool = False,
                 batch_size: Optional[int] = None):
        layout = tensor_layout or cls.tensor_layout()
        return _to_tensor(_pytorch_adapter,
                          layout,
                          obj or self,
                          dtype=cls._resolve_dtype(dtype),
                          batch=batch,
                          batch_size=batch_size)

    @classmethod
    def from_torch(cls,
                   tensor,
                   *,
                   tensor_layout: Optional[Type[TensorLayout]] = None,
                   batch: bool = False,
                   batch_size: Optional[int] = None):
        return _from_tensor(_pytorch_adapter,
                            tensor_layout or cls.tensor_layout(),
                            tensor,
                            batch=batch,
                            batch_size=batch_size)

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
    cls.to_numpy = hybridmethod(DataClassTensorMixin.to_numpy.__func__)
    cls.from_numpy = classmethod(DataClassTensorMixin.from_numpy.__func__)
    cls.to_torch = hybridmethod(DataClassTensorMixin.to_torch.__func__)
    cls.from_torch = classmethod(DataClassTensorMixin.from_torch.__func__)
    cls.tensor_layout = classmethod(DataClassTensorMixin.tensor_layout.__func__)
    cls._default_tensor_dtype = dtype
    cls._resolve_dtype = classmethod(DataClassTensorMixin._resolve_dtype.__func__)
    DataClassTensorMixin.register(cls)
    return cls

def config(shape: Optional[Iterable[int]]):
    return {"shape": shape}

def _to_tensor(adapter: TensorAdapter,
               layout: Type[TensorLayout],
               val,
               *,
               dtype="float",
               batch: bool = False,
               batch_size: Optional[int] = None):
    batch = batch or batch_size is not None
    shape = len(layout)
    if batch:
        batch_size = batch_size or (len(val) if hasattr(val, "__len__") else 0)
        if batch_size == 0:
            val = list(val)
            batch_size = len(val)
        shape = (batch_size, shape)
    tensor = adapter.zeros(shape, dtype=dtype)
    if not batch:
        layout.write(adapter, 0, tensor, val)
    else:
        for i, vi in enumerate(val):
            layout.write(adapter, 0, tensor[i], vi)
    return tensor

def _from_tensor(adapter: TensorAdapter,
                 layout: Type[TensorLayout],
                 tensor,
                 *,
                 batch: bool = False,
                 batch_size: Optional[int] = None):
    batch = batch or batch_size is not None
    if not batch:
        return layout.read(adapter, 0, tensor)
    batch_size = batch_size or (len(tensor) if hasattr(tensor, "__len__") else 0)
    result = [None]*batch_size
    if batch_size != 0:
        for i, t in enumerate(tensor):
            result[i] = layout.read(adapter, 0, t)
    else:
        for t in tensor:
            result.append(layout.read(adapter, 0, t))
    return result
