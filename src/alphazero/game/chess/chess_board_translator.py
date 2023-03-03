import torch
import chess

from game.board_translator import BoardTranslator

class ChessBoardTranslator(BoardTranslator):
    """Implements the board encoding from AlphaZero.
    Based on: https://github.com/iamlucaswolf/gym-chess/blob/6a5eb43650c400a556505ec035cc3a3c5792f8b2/gym_chess/alphazero/board_encoding.py

    This converts observations (chess.Board instances) to torch.Tensors using the encoding proposed in [Silver et al., 2017]

    An observation is a tensor of shape (k * 14 + 7, 8, 8). This is a 'stack' of 8x8 planes,
    where each of the k * 14 + 7 planes represents a different aspect of a game of chess.

    The first k * 14 planes encode the k most recent board positions, referred to as the board's history,
    grouped into 14 planes per position. The first six planes in the set represent the pieces of the
    active player (i.e. player for which the agent will move next). Each of the six planes is associated
    with a particular piece type (6 unique pieces). The next 6 encode pieces of the opposing player. The final two planes
    are binary, and indicate a two-fold and three-fold repetition of the encoded position over the course
    of the current game. In each stop, all board representations in the history are reoriented to the perspective
    of the active player.

    The remaining 7 planes encode meta-information about the game state:
    the color of the active player (0 = black, 1 = white), total move count, castling rights of player and opponent (king and queens-size),
    and the halfmove clock.

    Args:
        BoardTranslator (_type_): _description_
        
    """
    def __init__(self, k = 8):
        """Initialize this BoardTranslator instance.

        Args:
            k (int, optional): The number of recent board positions to be encoded. If the history from the current state < k, then the non-filled planes are zeroed out. Defaults to 8.
        """
        self.k = k

    def encode(self, board: chess.Board) -> torch.Tensor:
        history = torch.zeros(self.k, 14, 8, 8)
        move_number = board.fullmove_number

        # Copy the input board (we need to pop out the states to build up history)
        board = board.copy()

        # Push into history
        for i in range(min(self.k, move_number)):
            self.__push_into_history(board, history, i)

        # Concatenate planes and switch orientations
        history = self.__view(history, board.turn)

        # Add metadata
        meta = self.__meta(board)

        # Concatenate history and metadata
        ret = torch.cat([history, meta], dim = 0)

        return ret
    
    def __meta(self,
               board: chess.Board) -> torch.Tensor:
        meta = torch.zeros(7, 8, 8)

        # Active player color
        meta[0, :, :] = int(board.turn)

        # Total move count
        meta[1, :, :] = board.fullmove_number

        # Active player castling rights
        meta[2, :, :] = board.has_kingside_castling_rights(board.turn)
        meta[3, :, :] = board.has_queenside_castling_rights(board.turn)

        # Opponent player castling rights
        meta[4, :, :] = board.has_kingside_castling_rights(not board.turn)
        meta[5, :, :] = board.has_queenside_castling_rights(not board.turn)

        # No-progress counter
        meta[6, :, :] = board.halfmove_clock

        return meta
    
    def __view(self, 
               board: torch.Tensor, 
               orientation: chess.Color) -> torch.Tensor:
        if orientation == chess.BLACK:
            for board_array in board:
                # Rotate all planes encoding the position by 180 degrees
                rotated = torch.rot90(board_array[:12, :, :], k = 2)

                # In the buffer, the first six planes encode white's pieces;
                # Swap with the second six planes
                rotated = torch.roll(rotated, shift = 6, dims = 0)

                # Copy the first 12 elements of the third dimension back into the board array
                board_array[:12, :, :] = rotated
        

        # Concatenate k stacks of 14 planes to one stack of k * 14 planes
        board = torch.cat(torch.unbind(board, dim = 0), dim = 0)
        return board

    def __push_into_history(self, 
                            board: chess.Board, 
                            rep: torch.Tensor,
                            i: int) -> torch.Tensor:
        board_tensor = self.__encode_single_board(board)

        # Overwrite the oldest element in representation
        rep[i] = board_tensor

        # # Roll inserted element to the top (into the most recent position)
        # rep = torch.roll(rep, shifts = 1, dims = 0)

        return rep
        

    def __encode_single_board(self, board: chess.Board) -> torch.Tensor:
        tensor = torch.zeros(14, 8, 8)

        for square, piece in board.piece_map().items():
            rank, file = chess.square_rank(square), chess.square_file(square)
            piece_type, color = piece.piece_type, piece.color

            # The first 6 planes encode the pieces of the active player, and the following encode
            # the pieces of the opponent. 
            offset = 0 if color == chess.WHITE else 6

            # Chess enumerate piece types beginning with 1
            idx = piece_type - 1

            # Place into tensor
            tensor[idx + offset, rank, file] = 1 

        # Repetition counters
        tensor[12, :, :] = board.is_repetition(2)
        tensor[13, :, :] = board.is_repetition(3)

        return tensor

    def decode(self, board: torch.Tensor) -> chess.Board:
        # TODO
        pass