import chess
import torch

from game.game import Game

from .chess_board_translator import ChessBoardTranslator
from .chess_move_translator import ChessMoveTranslator

class ChessGame(Game):
    def __init__(self, 
                 device: torch.device
    ):
        super().__init__(device)
        self.board_translator = ChessBoardTranslator()
        self.move_translator = ChessMoveTranslator()

    def get_init_board(self) -> chess.Board:
        return chess.Board()
    
    def get_board_size(self) -> tuple:
        return (8, 8)
    
    def get_action_size(self):
        # 8 x 8 x 73
        return 4672

    def get_next_state(self, board: chess.Board, a: int) -> chess.Board:
        move = self.move_translator.decode(a, board)
        board.push(move)
        return board
    
    def get_valid_moves(self, board: chess.Board):
        valid_moves = board.legal_moves
        return torch.Tensor([self.move_translator.encode(move, board) for move in valid_moves])

    def get_game_ended(self, board):
        return board.is_game_over()

    def get_rewards(self, board: chess.Board):
        if self.getGameEnded(board):
            if board.result() == "1-0": # white wins
                return 1
            elif board.result() == "0-1": # black wins
                return -1
            else:
                return 0
        return None
    
    def string_representation(self, board) -> str:
        """Return the fen string representing this board.

        Args:
            board (_type_): board satte

        Returns:
            str: fen representation of the board
        """
        return board.fen()
    
    def get_current_player(self, board: chess.Board) -> bool:
        return board.turn
    
    # Translation methods
    def get_tensor_representation_of_board(self, board: chess.Board) -> torch.Tensor:
        return self.board_translator.encode(board)
    
    def get_integer_representation_of_move(self, move: chess.Move) -> torch.Tensor:
        return self.move_translator.encode(move)
    
    def get_board_from_tensor_representation(self, tensor: torch.Tensor) -> chess.Board:
        return self.board_translator.decode(tensor)
    
    def get_move_from_integer_representation(self, integer: int) -> chess.Move:
        return self.move_translator.decode(integer)
    