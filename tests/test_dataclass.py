import pytest

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

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
    num_moves: int
    next_move: Player
    board: List[Optional[Piece]] = field(metadata=config(shape=(64,)))

def test_dataclass_api():
    s1 = Chess(10, Player.BLACK, [Piece(PieceType.KING, Player.WHITE)])
    tensor = s1.to_numpy()
    assert tensor.shape == (579,)
    s2 = Chess.from_numpy(tensor)
    assert s2.next_move == Player.BLACK
    assert len(s2.board) == 64
    assert s2.board[0] == Piece(PieceType.KING, Player.WHITE)
    assert s2.board[1] is None
    assert len(set(s2.board[1:])) == 1


def test_unsupported_type_failure():
    @dataclass_tensor
    @dataclass
    class Unsupported:
        movie: str

    with pytest.raises(ValueError):
        Unsupported.tensor_layout()
