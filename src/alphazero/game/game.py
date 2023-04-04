import torch

from game.board_translator import BoardTranslator
from game.move_translator import MoveTranslator
from abc import ABC, abstractmethod

class Game:
    def __init__(self, 
                 device: torch.device
    ):
        self.device = device

    @abstractmethod
    def get_init_board(self) -> any:
        pass

    @abstractmethod
    def get_board_size(self) -> tuple:
        pass

    @abstractmethod
    def get_action_size(self) -> int:
        pass

    @abstractmethod
    def get_next_state(self, board, action) -> any:
        pass

    @abstractmethod
    def get_valid_moves(self, board) -> list[torch.Tensor]:
        pass
    
    @abstractmethod
    def get_game_ended(self, board) -> bool:
        pass

    @abstractmethod
    def get_rewards(self, board) -> int:
        pass

    @abstractmethod
    def string_representation(self, board) -> str:
        pass

    @abstractmethod
    def get_current_player(self, board) -> bool:
        pass

    @abstractmethod
    def get_tensor_representation_of_board(self, board) -> torch.Tensor:
        pass

    @abstractmethod
    def get_integer_representation_of_move(self, move) -> torch.Tensor:
        pass

    @abstractmethod
    def get_board_from_tensor_representation(self, tensor) -> any:
        pass

    @abstractmethod
    def get_move_from_integer_representation(self, tensor) -> any:
        pass

    # @abstractmethod
    # def get_canonical_form(self, board, player) -> any:
    #     pass

    # @abstractmethod
    # def get_symmetries(self, board, pi) -> list[torch.Tensor]:
    #     pass
