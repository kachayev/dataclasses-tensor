# Dataclasses Tensor

The library provides a simple API for encoding and decoding Python [`dataclasses`](https://docs.python.org/3/library/dataclasses.html) to and from tensors (PyTorch, TensorFlow, or NumPy arrays) based on `typing` annotations.

Heavily inspired by [`dataclasses-json`](https://github.com/lidatong/dataclasses-json) package.

## Install

```shell
pip install dataclasses-tensor
```

## Quickstart

Tensor representation for a game state in Chess:

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List

from dataclasses_tensor import dataclass_tensor, config

class Player(Enum):
  WHITE = 0
  BLACK = 1

class PieceType(Enum):
  PAWN = 0
  BISHOP = 1
  KNIGHT = 2
  ROOK = 3
  QUEEN = 4
  KING = 5

@dataclass
class Piece:
  piece_type: PieceType
  owner: Player

@dataclass_tensor
@dataclass
class Chess:
  num_moves: float
  next_move: Player
  board: List[Optional[Piece]] = field(metadata=config(shape=(64,)))
```

Working with tensors:

```python
>>> state = Chess(100., next_move=Player.WHITE, board=[Piece(PieceType.KING, Player.BLACK)])
>>> t1 = state.to_numpy()
array([100.,   1.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   1.,   0.,
         1.,   1.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   1.,
...
>>> t1.shape
(579,)
>>> Chess.from_numpy(t1)
Chess(num_moves=100., next_move=<Player.WHITE: 0>, board=[Piece(piece_type=<PieceType.KING: 5>, owner=<Player.BLACK: 1>), ...])
```

## Types

### Data Classes

The library uses type annotations to determine appropriate encoding layout. Data class member variables serialized sequentially. See supported types listed below. 

### Primitives (int, float, bool)

The library supports numerical primitives (`int`, `float`) and `bool`. Strings and byte arrays are not supported.

Warning: be careful with tensor `dtype` as an implicit type conversion could potentially lead to losing information (for example, writing `float` into `int32` tensor and reading it back won't produce expected result).

### Enums

Python [`Enums`](https://docs.python.org/3/library/enum.html) are encoded using one-hot encoding.

```python
>>> from dataclasses_tensor import dataclass_tensor
>>> from dataclasses import dataclass
>>> from enum import Enum
>>>
>>> class Matrix(Enum):
...     THE_MATRIX = 1
...     RELOADED = 2
...     REVOLUTIONS = 3
...
>>> @dataclass_tensor
... @dataclass
... class WatchList:
...     matrix: Matrix
...
>>> WatchList(Matrix.RELOADED).to_numpy()
array([0., 0., 1.])
>>> WatchList.from_numpy(_)
WatchList(matrix=<Matrix.RELOADED: 2>)
```

### Optional

[`typing.Optional`](https://docs.python.org/3/library/typing.html#typing.Optional) type is encoded using additional dimension prior to the main datatype.

```python
>>> from typing import Optional
>>>
>>> @dataclass_tensor
... @dataclass
... class MaybeWatchList:
...     matrix: Optional[Matrix]
>>>
>>> MaybeWatchList(Matrix.RELOADED).to_numpy()
array([0., 0., 1., 0.])
>>> MaybeWatchList.from_numpy([0., 0., 1., 0.])
MaybeWatchList(matrix=<Matrix.RELOADED: 2>)
>>> MaybeWatchList.from_numpy([1., 0., 0., 0.])
MaybeWatchList(matrix=None)
```

The layout described for `Optional[Enum]` is consistent with having `None` as additional option into enumeration.

### Arrays

Arrays, defined either using [`typing.List`](https://docs.python.org/3/library/typing.html#typing.List) or `[]` (supported in Python3.9+), require size to be statically provided. See example:

```python
>>> from typing import List
>>> from dataclasses_tensor import config

>>> @dataclass_tensor
... @dataclass
... class MultipleWatchList:
...     matrices: List[Matrix] = field(metadata=config(shape=(2,)))
>>>
>>> MultipleWatchList([Matrix.THE_MATRIX, Matrix.RELOADED]).to_numpy()
array([1., 0., 0., 0., 1., 0.])
>>> MultipleWatchList.from_numpy([1., 0., 0., 0., 1., 0.])
MultipleWatchList(matrices=[<Matrix.THE_MATRIX: 1>, <Matrix.RELOADED: 2>])
```

Nested lists are supported, note multidimensional `shape` configuration:

```python
>>> @dataclass_tensor
... @dataclass
... class MultipleWatchList:
...     matrices: List[List[Matrix]] = field(metadata=config(shape=(1,2)))
>>>
>>> MultipleWatchList([[Matrix.THE_MATRIX, Matrix.RELOADED]]).to_numpy()
array([1., 0., 0., 0., 1., 0.])
>>> MultipleWatchList.from_numpy([1., 0., 0., 0., 1., 0.])
MultipleWatchList(matrices=[[<Matrix.THE_MATRIX: 1>, <Matrix.RELOADED: 2>]])
```

If `List` argument is `Optional`, the list is automatically padded to the right shape with `None`s.

```python
>>> @dataclass_tensor
... @dataclass
... class MaybeMultipleWatchList:
...     matrices: List[Optional[Matrix]] = field(metadata=config(shape=(3,)))
>>>
>>> MaybeMultipleWatchList([Matrix.THE_MATRIX, Matrix.RELOADED]).to_numpy()
array([0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0.])
>>> MaybeMultipleWatchList.from_numpy([0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0.])
MaybeMultipleWatchList(matrices=[<Matrix.THE_MATRIX: 1>, <Matrix.RELOADED: 2>, None])
```

### Union

[`typing.Union`]() is encoded by allocating one-hot tensor to determine which option from the union is given following by corresponding layouts for all options.

```python
>>> from typing import Union
>>>
>>> class Batman(Enum):
...     BEGINS = 1
...     DARK_KNIGHT = 2
...     DARK_KINGHT_RISES = 3
...
>>> @dataclass_tensor
... @dataclass
... class WatchList:
...     next_movie: Union[Matrix, Batman]
...
>>> WatchList(Matrix.RELOADED).to_numpy()
array([1., 0., 0., 1., 0., 0., 0., 0.])
>>> WatchList.from_numpy(_)
WatchList(next_movie=<Matrix.RELOADED: 2>)
>>> WatchList(Batman.DARK_KNIGHT).to_numpy()
array([0., 1., 0., 0., 0., 0., 1., 0.])
>>> WatchList.from_numpy(_)
WatchList(next_movie=<Batman.DARK_KNIGHT: 2>)
```

Decoding is a fairly straigtforward process though encoding might be somewhat problematic: Python's `typing` is not designed to provide separation-by-construction for union types. The library uses simple `isinstance` checks to test out all types provided against a given value, first match is used. The library does not traverse generics, origins, supertypes, etc. So, be diligent defining of `Union`. 

### Recursive Definitions

Recursive definitions, like linked lists, trees, graphs etc, are **not supported**. From a usability and performance point of view, it's crucial for encoder/decoder to be able to evaluate statically output tensor size.

## Targets

The library supports the following containers as tensors:

* [NumPy ndarray](https://numpy.org/doc/stable/reference/generated/numpy.array.html) with `to_numpy`/`from_numpy`
* [PyTorch tensors](https://pytorch.org/docs/stable/tensors.html) with `to_torch`/`from_torch`
* [TensorFlow tensors](https://www.tensorflow.org/api_docs/python/tf/Tensor) with `to_tf`/`from_tf`

Note, that dependencies are not installed with the library itself (TensorFlow, PyTorch or NumPy) and should be provided at runtime.

## Performance

Tensor layout is not cached and is computed for each operation. When performing a lot of operations with class definition staying the same, it makes sense to re-use layout. For example:

```python
>>> class Matrix(Enum):
...     THE_MATRIX = 1
...     RELOADED = 2
...     REVOLUTIONS = 3
...
>>> @dataclass_tensor
... @dataclass
... class WatchList:
...     matrix: Matrix
...
>>> layout = WatchList.tensor_layout()
>>> WatchList(Matrix.RELOADED).to_numpy(tensor_layout=layout)
array([0., 0., 1.])
>>> WatchList.from_numpy(_, tensor_layout=layout)
WatchList(matrix=<Matrix.RELOADED: 2>)
```

## Advanced Features

### Dtype

The library supports float and integer (long) tensors. The data type could be specified either as a parameter to the `dataclass_tensor` decorator (applied to all operations) or independently as an argument to `to_tensor` function call. See examples below.

`dtype` argument is passed to the corresponding target library, e.g. NumPy ([docs](https://numpy.org/doc/stable/reference/arrays.dtypes.html)), PyTorch ([docs](https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.dtype)) or TensorFlow.

```python
>>> class Matrix(Enum):
...     THE_MATRIX = 1
...     RELOADED = 2
...     REVOLUTIONS = 3
...
>>> @dataclass_tensor
... @dataclass
... class WatchList:
...     matrix: Matrix
...
>>> WatchList(Matrix.RELOADED).to_numpy()
array([0., 0., 1.], dtype=float32)
>>> WatchList(Matrix.RELOADED).to_numpy(dtype="int32")
array([0, 0, 1], dtype=int32)
```

or with defaults setup in a decorator

```python
>>> class Matrix(Enum):
...     THE_MATRIX = 1
...     RELOADED = 2
...     REVOLUTIONS = 3
...
>>> @dataclass_tensor(dtype="int32")
... @dataclass
... class WatchList:
...     matrix: Matrix
...
>>> WatchList(Matrix.RELOADED).to_numpy()
array([0, 0, 1], dtype=int32)
```

### Custom Attribute Resolver

TBD

### Batch

TBD

## TODO

- [ ] Tests suite for PyTorch and TensorFlow adapters
- [ ] Custom attribute resolver (e.g. from dict instead of class instance)
- [ ] Batch operations (write many/read many)
- [ ] Pretty-print for tensor layout object

## Contributing

* Check for open issues or open a fresh issue to start a discussion around a feature idea or a bug.
* Fork the repository on Github & branch from `main` to `feature-*` to start making your changes.
* Write a test which shows that the bug was fixed or that the feature works as expected.

or simply...

* Use it.
* Enjoy it.
* Spread the word.

## License

Copyright © 2021, Oleksii Kachaiev.

`dataclasses-tensor` is licensed under the MIT license, available at MIT and also in the LICENSE file.
