from dataclasses import dataclass
from enum import Enum

from dataclasses_tensor import dataclass_tensor
import numpy as np

class Movie(Enum):
    THE_MATRIX = 0
    THE_DARK_KNIGHT = 1
    INTERSTELLAR = 2

@dataclass_tensor
@dataclass
class Watch:
    next_movie: Movie

def test_numpy_dtype_argument():
    s1 = Watch(Movie.THE_DARK_KNIGHT)
    t1 = s1.to_numpy(dtype="int32")
    assert s1 == Watch.from_numpy(t1)
    assert t1.dtype == np.int32
    t2 = s1.to_numpy(dtype="float32")
    assert s1 == Watch.from_numpy(t2)
    assert t2.dtype == np.float32
    t3 = s1.to_numpy(dtype=np.int8)
    assert s1 == Watch.from_numpy(t3)
    assert t3.dtype == np.int8

def test_numpy_dtype_class():
    @dataclass_tensor(dtype="float32")
    @dataclass
    class WatchDtype:
        next_movie: Movie

    s1 = WatchDtype(Movie.THE_MATRIX)
    t1 = s1.to_numpy()
    assert s1 == WatchDtype.from_numpy(t1)
    assert t1.dtype == np.float32

