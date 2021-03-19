from typing import Union

class TensorAdapter:
    def zeros(self, size: int, dtype: str):
        raise NotImplemented()
    
    def argmax(self, arr):
        raise NotImplemented()

    def get(self, tensor, pos):
        raise NotImplemented()

try:
    import numpy as np
    class NumpyAdapter(TensorAdapter):
        def zeros(self, size: int, dtype: Union[str, 'np.dtype']):
            return np.zeros(size, dtype=dtype)

        def argmax(self, arr):
            return np.argmax(arr)

        def get(self, arr, pos):
            return arr[pos]
except ImportError:
    class NumpyAdapter(TensorAdapter):
        def zero(self, _size: int, _dtype: str):
            raise RuntimeError("numpy library is not installed")
        
        def argmax(self, _arr):
            raise RuntimeError("numpy library is not installed")
        
        def get(self, _arr, _pos):
            raise RuntimeError("numpy library is not installed")

_numpy_adapter = NumpyAdapter()

try:
    import torch
    class PyTorchAdapter(TensorAdapter):
        def zeros(self, size: int, dtype: Union[str, 'torch.dtype']):
            if isinstance(dtype, str):
                dtype = torch.__getattribute__(dtype)
            return torch.zeros(size, dtype=dtype)

        def argmax(self, arr):
            return torch.argmax(arr)

        def get(self, arr, pos):
            return arr[pos].item()
except ImportError:
    class PyTorchAdapter(TensorAdapter):
        def zero(self, _size: int, _dtype: str):
            raise RuntimeError("torch library is not installed")
        
        def argmax(self, _arr):
            raise RuntimeError("torch library is not installed")

        def get(self, _arr, _pos):
            raise RuntimeError("torch library is not installed")

_pytorch_adapter = PyTorchAdapter()

try:
    import tensorflow as tf
    class TensorFlowAdapter(TensorAdapter):
        def zeros(self, size: int, dtype: str):
            return tf.zeros(size)

        def argmax(self, arr):
            return tf.argmax(arr)

        def get(self, arr, pos):
            return arr[pos]
except ImportError:
    class TensorFlowAdapter(TensorAdapter):
        def zero(self, _size: int, _dtype: str):
            raise RuntimeError("tensorflow library is not installed")
        
        def argmax(self, _arr):
            raise RuntimeError("tensorflow library is not installed")

        def get(self, _arr, _pos):
            raise RuntimeError("tensorflow library is not installed")

_tf_adapter = TensorFlowAdapter()
