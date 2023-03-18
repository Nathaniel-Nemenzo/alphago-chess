import torch

from game.board_translator import BoardTranslator
from game.move_translator import MoveTranslator
from abc import ABC, abstractmethod

class Game:
    def __init__(self, 
                 device: torch.device,
                 board_translator: BoardTranslator,
                 move_translator: MoveTranslator):
        self.device = device
        self.board_translator = board_translator
        self.move_translator = move_translator

    @abstractmethod
    def getInitBoard(self) -> any:
        pass

    @abstractmethod
    def getBoardSize(self) -> tuple:
        pass

    @abstractmethod
    def getActionSize(self) -> int:
        pass

    @abstractmethod
    def getNextState(self, board, action) -> any:
        pass

    @abstractmethod
    def getValidMoves(self, board) -> list[torch.Tensor]:
        pass
    
    @abstractmethod
    def getGameEnded(self, board) -> bool:
        pass

    @abstractmethod
    def getResult(self, board) -> bool:
        pass

    # @abstractmethod
    # def getCanonicalForm(self, board, player) -> any:
    #     pass

    @abstractmethod
    def getRewards(self, board) -> int:
        pass

    @abstractmethod
    def getSymmetries(self, board, pi) -> list[torch.Tensor]:
        pass

    @abstractmethod
    def stringRepresentation(self, board) -> str:
        pass

    @abstractmethod
    def checkIfValid(self, board, action) -> bool:
        pass

    @abstractmethod
    def getCurrentPlayer(self, board) -> bool:
        pass