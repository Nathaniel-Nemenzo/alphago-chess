"""
Encapsulates classes for Board encoding and decoding in AlphaZero board representation
"""

import torch
import chess

class BoardRepresentation:
    """
    Implements the board encoding from AlphaZero.
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
        k: The number of recent board positions encoded in an observation (corresponds to the 'k' parameter above).

    Observations:
        Tensor(k * 14 + 7, 8, 8) <- pytorch convention is (N, C, W, H)
    """


    def __init__(self, k = 8):
        self._history = BoardHistory(k)


    def reset(self):
        """
        Clears board history.
        """
        self._history.reset()

    def observation(self, board):
        """
        Stores the observation in the representation and converts chess.Board observations instance to Tensors
        """
        self._history.push(board)
        history = self._history.view(orientation = board.turn)
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

        observation = torch.cat([history, meta], dim = 0)
        return observation

class BoardHistory:
    """
    Maintains a history of recent board positions, encoded as Tensors. The history only retains the k most recent board positions;
    older positions are discarded when new ones are added. The BoardHistory class uses LIFO conventions, where positions at the front of the buffer are
    more recent.

    The BoardHistory class always stores boards oriented towards player 1 (white).

    Args:
        k: The number of most recent board positions to save.
    """

    def __init__(self, k):
        self.k = k
        self._buffer = torch.zeros(k, 14, 8, 8)

    def push(self, board):
        """
        Push a position into the history. If the size after adding > k, then the oldest position is discarded. Positions are stored
        using the Tensor encoding suggested in [Silver et al., 2017]
        
        Args:
            board: The position to be added to the history
        """
        board_tensor = self.encode(board)

        # Overwrite the oldest element in the buffer
        self._buffer[-1] = board_tensor

        # Roll inserted element to the top (into the most recent position)
        self._buffer = torch.roll(self._buffer, shifts = 1, dims = 0)

    def encode(self, board):
        """
        Encode a chess.Board instance to Tensor representation as described above.

        Args:
            board: The chess.Board instance to encode
        """
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

    def view(self, orientation):
        """
        Return a Tensor(k * 14, 8, 8) representation of this BoardHistory instance. If less than k positions have been added since the last reset
        or since instantiation, missing positions are zerod out.

        Positions are oriented toward player white by default; setting the optional orientation parameter to 'chess.BLACK' will reorient
        the view toward player black.

        Args:
            orientation: Boolean representing 0 = black and 1 = white
        """
        ret = self._buffer.clone()

        if orientation == chess.BLACK:
            for board_array in ret:
                # Rotate all planes encoding the position by 180 degrees
                rotated = torch.rot90(board_array[:12, :, :], k = 2)

                # In the buffer, the first six planes encode white's pieces;
                # Swap with the second six planes
                rotated = torch.roll(rotated, shift = 6, dims = 0)

                # Copy the first 12 elements of the third dimension back into the board array
                board_array[:12, :, :] = rotated

        # Concatenate k stacks of 14 planes to one stack of k * 14 planes
        ret = torch.cat(torch.unbind(ret, dim = 0), dim = 0)
        return ret

    def reset(self):
        """
        Clear the history of the buffer.
        """
        self._buffer[:] = 0

if __name__ == "__main__":
    rep = BoardRepresentation()
    board = chess.Board()
    observation = rep.observation(board)
    print(observation[0])