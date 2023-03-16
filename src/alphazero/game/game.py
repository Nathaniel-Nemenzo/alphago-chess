import torch

from abc import ABC, abstractmethod

class Game:
    def __init__(self):
        pass

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