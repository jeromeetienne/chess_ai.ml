from typing import List, Optional, Protocol, runtime_checkable

@runtime_checkable
class GameProtocol(Protocol):
    def get_legal_moves(self) -> List[int]:
        ...

    def get_game_state(self) -> List[int]:
        ...

    def make_move(self, move: int) -> "GameProtocol":
        ...

    def check_win(self) -> Optional[int]:
        ...

    def is_game_over(self) -> bool:
        ...
