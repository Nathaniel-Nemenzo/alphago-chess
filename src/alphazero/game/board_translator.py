import torch

from abc import ABC, abstractmethod

class BoardTranslator(ABC):
    """ Class that translates a user/library defined class (like chess.Board) into a tensor according to some scheme, (like 119 x 8 x 8 tensors in [Silver et al.])

    Args:
        ABC (_type_): _description_
    """
    def __init__(self):
        pass

    @abstractmethod
    def encode(self, board: any) -> torch.Tensor:
        """ Encode a user/library defined board into a tensor.

        Args:
            move (any): 

        Returns:
            torch.Tensor: Tensor representing the board state
        """
        pass

    @abstractmethod
    def decode(self, board: torch.Tensor) -> any:
        """ Decode a board represented as a tensor into the user/library defined board instance

        Args:
            board (torch.Tensor): 

        Returns:
            any: User/library defined class
        """
        pass