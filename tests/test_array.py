import pytest

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from dataclasses_tensor import config, dataclass_tensor

class Movie(Enum):
    THE_MATRIX = 0
    THE_TWO_TOWERS = 1

@dataclass_tensor
@dataclass
class Watch:
    movies: List[Movie] = field(metadata=config(shape=(2,)))

@dataclass_tensor
@dataclass
class WatchOptional:
    movies: List[Optional[Movie]] = field(metadata=config(shape=(2,)))

@dataclass_tensor
@dataclass
class WatchMore:
    movies: List[List[Movie]] = field(metadata=config(shape=(2,2)))

@dataclass_tensor
@dataclass
class WatchOneDim:
    movies: List[Movie] = field(metadata=config(shape=2))

def test_basic_array():
    s1 = Watch(movies=[Movie.THE_MATRIX, Movie.THE_TWO_TOWERS])
    t1 = s1.to_numpy()
    assert Watch.from_numpy(t1) == s1

def test_optional_array():
    s1 = WatchOptional(movies=[Movie.THE_MATRIX, Movie.THE_TWO_TOWERS])
    t1 = s1.to_numpy()
    assert WatchOptional.from_numpy(t1) == s1

    s2 = WatchOptional(movies=[None, Movie.THE_TWO_TOWERS])
    t2 = s2.to_numpy()
    assert WatchOptional.from_numpy(t2) == s2

def test_optional_padding():
    s1 = WatchOptional(movies=[Movie.THE_MATRIX])
    t1 = s1.to_numpy()
    s2 = WatchOptional.from_numpy(t1)
    assert len(s2.movies) == 2
    assert s2.movies[1] is None

def test_non_optional_padding_failure():
    with pytest.raises(ValueError):
        Watch(movies=[Movie.THE_MATRIX]).to_numpy()

def test_multidimensional_array():
    s1 = WatchMore(movies=[
        [Movie.THE_MATRIX, Movie.THE_MATRIX],
        [Movie.THE_TWO_TOWERS, Movie.THE_TWO_TOWERS]
    ])
    t1 = s1.to_numpy()
    assert s1 == WatchMore.from_numpy(t1)

def test_single_dimension_shape():
    s1 = WatchOneDim(movies=[Movie.THE_MATRIX, Movie.THE_TWO_TOWERS])
    t1 = s1.to_numpy()
    assert WatchOneDim.from_numpy(t1) == s1
