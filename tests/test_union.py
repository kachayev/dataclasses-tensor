import pytest

from dataclasses import dataclass
from enum import Enum
from typing import Union

from dataclasses_tensor import dataclass_tensor

class Matrix(Enum):
    THE_MATRIX = 0
    THE_RELOADED = 1
    THE_REVOLUTIONS = 2

class Rings(Enum):
    THE_FELLOWSHIP = 0
    THE_TWO_TOWERS = 1
    THE_RETURN = 2

class Batman(Enum):
    BEGINS = 0
    DARK_KNIGHT = 1
    DARK_KNIGHT_RISES = 2

class Watched(Enum):
    I_AM_LEGEND = 0

@dataclass_tensor
@dataclass
class Watch:
    next_movie: Union[Matrix, Rings, Batman]

def test_basic_union():
    s1 = Watch(Matrix.THE_RELOADED)
    t1 = s1.to_numpy()
    print(t1)
    assert Watch.from_numpy(t1) == s1

    s2 = Watch(Rings.THE_FELLOWSHIP)
    t2 = s2.to_numpy()
    assert Watch.from_numpy(t2) == s2

    s3 = Watch(Batman.DARK_KNIGHT_RISES)
    t3 = s3.to_numpy()
    assert Watch.from_numpy(t3) == s3

def test_invalid_value_failure():
    with pytest.raises(ValueError):
        Watch(Watched.I_AM_LEGEND).to_numpy()

