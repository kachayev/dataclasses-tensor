from dataclasses import dataclass
from enum import Enum, auto

from dataclasses_tensor import dataclass_tensor

def test_basic_enum():
    class Movie(Enum):
        THE_MATRIX = 0
        THE_TWO_TOWERS = 1

    @dataclass_tensor
    @dataclass
    class Watch:
        next_movie: Movie

    s1 = Watch(next_movie=Movie.THE_MATRIX)
    t1 = s1.to_numpy()
    assert Watch.from_numpy(t1) == s1
    
    s2 = Watch(next_movie=Movie.THE_TWO_TOWERS)
    t2 = s2.to_numpy()
    assert Watch.from_numpy(t2) == s2

def test_auto_enum():
    class MovieAuto(Enum):
        THE_MATRIX = auto()
        THE_TWO_TOWERS = auto()
        THE_DARK_KNIGHT = auto()

    @dataclass_tensor
    @dataclass
    class WatchAuto:
        next_movie: MovieAuto

    s1 = WatchAuto(next_movie=MovieAuto.THE_MATRIX)
    t1 = s1.to_numpy()
    assert WatchAuto.from_numpy(t1) == s1
    
    s2 = WatchAuto(next_movie=MovieAuto.THE_TWO_TOWERS)
    t2 = s2.to_numpy()
    assert WatchAuto.from_numpy(t2) == s2
