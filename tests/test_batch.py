from dataclasses import dataclass
from enum import Enum

from dataclasses_tensor import dataclass_tensor

class Movie(Enum):
    THE_MATRIX = 0
    THE_DARK_KNIGHT = 1
    INTERSTELLAR = 2

@dataclass_tensor
@dataclass
class Watch:
    next_movie: Movie

def test_batch_api():
    s1 = Watch(Movie.THE_MATRIX)
    s2 = Watch(Movie.THE_DARK_KNIGHT)
    s3 = Watch(Movie.INTERSTELLAR)
    b = Watch.to_numpy([s1, s2, s3], batch=True)
    assert b.shape == (3,3)
    [sr1, sr2, sr3] = Watch.from_numpy(b, batch=True)
    assert s1 == sr1
    assert s2 == sr2
    assert s3 == sr3

def test_batch_size():
    s1 = Watch(Movie.THE_MATRIX)
    s2 = Watch(Movie.THE_DARK_KNIGHT)
    b = Watch.to_numpy([s1, s2], batch_size=2)
    assert b.shape == (2,3)
    [sr1, sr2] = Watch.from_numpy(b, batch_size=2)
    assert s1 == sr1
    assert s2 == sr2

def test_batch_no_size():
    s1 = Watch(Movie.THE_MATRIX)
    s2 = Watch(Movie.THE_DARK_KNIGHT)

    def generate_batch():
        yield s1
        yield s2

    b = Watch.to_numpy(generate_batch(), batch=True)
    assert b.shape == (2,3)
    [sr1, sr2] = Watch.from_numpy(b, batch=True)
    assert s1 == sr1
    assert s2 == sr2
