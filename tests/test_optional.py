from dataclasses import dataclass
from enum import Enum
from typing import Optional

from dataclasses_tensor import dataclass_tensor

def test_basic_optional():
    class Movie(Enum):
        THE_MATRIX = 0
        THE_TWO_TOWERS = 1

    @dataclass_tensor
    @dataclass
    class Watch:
        next_movie: Optional[Movie]

    s1 = Watch(next_movie=Movie.THE_MATRIX)
    t1 = s1.to_numpy()
    assert Watch.from_numpy(t1) == s1
    
    s2 = Watch(next_movie=None)
    t2 = s2.to_numpy()
    assert Watch.from_numpy(t2) == s2
