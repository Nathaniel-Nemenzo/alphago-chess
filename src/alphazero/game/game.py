import torch

from abc import ABC, abstractmethod

class Game:
    def __init__(self):
        pass

    @abstractmethod
    def getInitBoard(self):
        pass

    @abstractmethod
    def getBoardSize(self):
        pass

    @abstractmethod
    def getActionSize(self):
        pass

    @abstractmethod
    def getNextState(self, board, player, action):
        pass

    @abstractmethod
    def getValidMoves(self, board, player):
        pass
    
    @abstractmethod
    def getGameEnded(self, board, player):
        pass

    @abstractmethod
    def getCanonicalForm(self, board, player):
        pass

    @abstractmethod
    def getSymmetries(self, board, pi):
        pass

    @abstractmethod
    def stringRepresentation(self, board):
        pass