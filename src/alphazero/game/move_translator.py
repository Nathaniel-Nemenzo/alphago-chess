from abc import ABC, abstractmethod

import chess
import torch
import numpy as np

class MoveTranslator(ABC):
    """
    Abstract class for converting to/from move representations (as specified by user or libraries like chess.Move) and integers that represent the action, like how actions are encoded in [Silver et al.] for chess in a (1, 4672) tensor.

    Moves are always encoded from the perspective of the first player (e.g. white in chess).
    """
    def __init__(self, device: torch.device):
        self.device = device

    @abstractmethod
    def encode(self, move: any, board: any) -> int:
        """ Encode a game move of any type to an integer.

        Args:
            move (any): Game move
            board (any): Board keeping track of game state

        Returns:
            int: Integer representing game move
        """
        pass

    @abstractmethod
    def decode(self, move: int, board: any) -> any:
        """ Decode a move of any type (as encoded by self.encode()) to its corresponding move

        Args:
            move (int): Integer to decode
            board (any): Board keeping track of game state

        Returns:
            any: Game move
        """
        pass
