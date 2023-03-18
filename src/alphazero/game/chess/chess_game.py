import chess
import torch

from game.game import Game

from .chess_board_translator import ChessBoardTranslator
from .chess_move_translator import ChessMoveTranslator

class ChessGame(Game):
    def __init__(self, 
                 device: torch.device,
                 board_translator: ChessBoardTranslator,
                 move_translator: ChessMoveTranslator):
        super().__init__(device, board_translator, move_translator)

    def getInitBoard(self) -> chess.Board:
        """Gets a new board (creates a new Chess game)

        Returns:
            _type_: 'Default' chess.Board instance
        """
        return chess.Board()
    
    def getBoardSize(self) -> tuple:
        return (8, 8)
    
    def getActionSize(self):
        """Returns the action size a chess board. 

        Returns:
            _type_: Size of action space of the chess board. Corresponds to 8 x 8 x 73, where 73 represents 56 queen moves, 8 knight moves, and 9 underpromotions.
        """
        # 8 x 8 x 73
        return 4672

    def getNextState(self, board: chess.Board, a: chess.Move) -> chess.Board:
        """Gets the next state of a board and a move

        Args:
            board (chess.Board): Board of which to get the next state
            a (int): Integer representing an action to take on the board

        Returns:
            chess.Board: State of the board after the action is taken
        """
        # Decide the move represents a pawn move to the edge ranks (queen promotion)
        move = self.move_translator.decode(a, board)
        board.push(move)
        return board
    
    def getValidMoves(self, board: chess.Board):
        """Returns the valid actions of the current valid player encoded into the (8 x 8 x 73) action space.

        Args:
            board (chess.Board): State of the board

        Returns:
            List[Tensor]: List representing the encoded states of the board
        """
        return list(board.legal_moves)

    def getGameEnded(self, board):
        """Returns whether or not a passed in game is over

        Args:
            board (_type_): Board to decide whether the game is over or not

        Returns:
            _type_: _description_
        """
        return board.is_game_over()
    
    def getResult(self, board: chess.Board) -> bool:
        return board.result()

    def getRewards(self, board: chess.Board):
        """Returns the rewards of a passed in terminal state. If the white player wins, a reward of 1 is returned. If the black player wins, a reward of 0 is returned.

        Args:
            board (_type_): Board instance to get the reward from

        Returns:
            _type_: 1 representing a win for white, -1 representing a win for black, and 0 representing a draw.
        """
        if self.getGameEnded(board):
            if board.result() == "1-0":
                # White won
                return 1 if board.turn == True else -1
            elif board.result() == "0-1":
                # Black won
                return -1 if board.turn == False else -1
            else:
                return 0
        return None
    
    def getSymmetries(self, board, pi):
        """Return an empty list. (chess has no symmetries)

        Args:
            board (_type_): _description_
            pi (_type_): _description_

        Returns:
            _type_: _description_
        """
        return []
    
    def stringRepresentation(self, board) -> str:
        """Return the fen string representing this board.

        Args:
            board (_type_): board satte

        Returns:
            str: fen representation of the board
        """
        return board.fen()
    
    def checkIfValid(self, board: chess.Board, action: chess.Move) -> bool:
        return board.is_legal(action)
    
    def getCurrentPlayer(self, board: chess.Board) -> bool:
        return board.turn