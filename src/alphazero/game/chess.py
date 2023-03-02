"""
Wrapper around the chess.Board class which provides data representation of boards and moves along with
other functionalities for AlphaZero
"""

import chess

from helpers.move import MoveRepresentation
from helpers.board import BoardRepresentation

move_encoder = MoveRepresentation()
board_encoder = BoardRepresentation()

class ChessGame():
    def gameEnded(self, board):
        return board.is_game_over()

    def gameRewards(self, board):
        if self.gameEnded(board):
            if board.result() == "1-0":
                # White won
                return 1 if board.turn == chess.BLACK else -1
            elif board.result() == "0-1":
                # Black won
                return 1 if board.turn == chess.WHITE else -1
            else:
                return 0

    def getValidActions(self, board: chess.Board):
        """
        Get all the valid actions for a certain board

        Returns:
            - List of chess actions encoded to [Silver et al.] representation
        """
        
        return [move_encoder.encode(move) for move in board.legal_moves if board.color_at(move.from_square) == board.turn]

    def nextState(self, board: chess.Board, a: int) -> chess.Board:
        """
        Apply an action to this game and get the resulting state

        Args:
            - a: Action to take specified as an integer according to [Silver et al.]
        """

        move = move_encoder.decode(a)
        board.push(move)

        # might leak reference??
        return board

    def getBoard(self, board):
        """
        Args:
            - board: board to get representation
        Returns:
            board: a representation of the board that is of model input form (119x8x8).
        
        Encodes board history
        """

        return board_encoder.observation(board)
    
    def newBoard(self):
        return chess.Board()