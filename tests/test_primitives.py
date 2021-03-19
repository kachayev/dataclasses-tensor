from dataclasses import dataclass

from dataclasses_tensor import dataclass_tensor

@dataclass_tensor
@dataclass
class Primitives:
    f: float
    i: int
    b: bool

def test_primitives():
    s1 = Primitives(10., 1, True)
    t1 = s1.to_numpy()
    assert s1 == Primitives.from_numpy(t1)
