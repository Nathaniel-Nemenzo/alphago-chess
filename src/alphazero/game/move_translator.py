from abc import ABC, abstractmethod

class MoveTranslator(ABC):
    """
    Abstract class for converting to/from move representations (as specified by user or libraries like chess.Move) and integers that represent the action, like how actions are encoded in [Silver et al.] for chess in a (1, 4672) tensor.
    """
    def __init__(self):
        pass

    @abstractmethod
    def encode(self, move: any) -> int:
        """ Encode a game move of any type to an integer.

        Args:
            move (any): Game move

        Returns:
            int: Integer representing game move
        """
        pass

    @abstractmethod
    def decode(self, move: int) -> any:
        """ Decode a move of any type (as encoded by self.encode()) to its corresponding move

        Args:
            move (int): Integer to decode

        Returns:
            any: Game move
        """
        pass