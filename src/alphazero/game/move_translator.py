from abc import ABC, abstractmethod

import chess
import numpy as np

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

class QueenMovesEncoding:
    def __init__(self):
        self._TYPE_OFFSET = 0
        self._NUM_TYPES = 56
        self._DIRECTIONS = utils.IndexedTuple(
            (0, 1),  # N
            (1, 1),  # NE
            (1, 0),  # E
            (1, -1),  # SE
            (0, -1),  # S
            (-1, -1),  # SW
            (-1, 0),  # W
            (-1, 1)  # NW
        )

    def encode(self, move):
        """
        Encode a queen move into action index representation, if possible, else returns None
        """
        from_rank, from_file, to_rank, to_file = utils.unpack(move)
        delta = (to_rank - from_rank, to_file - from_file)

        # Determine whether horizontal, vertical, diagonal, queen move promotion
        is_horizontal = delta[0] == 0
        is_vertical = delta[1] == 0
        is_diagonal = abs(delta[0]) == abs(delta[1])
        is_promotion = move.promotion in (chess.QUEEN, None) # True for promotion to queen and no promotion, else false (underpromotion)

        is_queen_move = ((is_horizontal or is_vertical or is_diagonal) and is_promotion)
        if not is_queen_move:
            return None
        
        direction = tuple(np.sign(delta))
        distance = np.max(np.abs(delta))

        direction_idx = self._DIRECTIONS.index(direction)
        distance_idx = distance - 1

        move_type = np.ravel_multi_index(multi_index = ([direction_idx, distance_idx]), dims = (8, 7))
        action = np.ravel_multi_index(multi_index = ((move_type, from_rank, from_file)), dims = (73, 8, 8))
        return action

    def decode(self, index):
        """
        Decode a moFve in action index representation to a queen move
        """
        move_type, from_rank, from_file = np.unravel_index(index, (73, 8, 8))
        is_queen_move = move_type < self._NUM_TYPES
        if not is_queen_move:
            return None
        
        # Get original direction and distance
        direction_idx, distance_idx = np.unravel_index(
            indices = move_type,
            shape = (8, 7)
        )
        
        # Get the direction
        direction = self._DIRECTIONS[direction_idx]
        distance = distance_idx + 1

        # Get deltas (for queen moves, it can be either diagonal (components match), or horizontal / vertical (other component is 0))
        delta_rank = direction[0] * distance
        delta_file = direction[1] * distance

        to_rank = from_rank + delta_rank
        to_file = from_file + delta_file

        move = utils.pack(from_rank, from_file, to_rank, to_file)
        return move

class KnightMovesEncoding:
    def __init__(self):
        self._TYPE_OFFSET = 56
        self._NUM_TYPES = 8
        self._DIRECTIONS = utils.IndexedTuple(
            (2, 1),
            (1, 2),
            (-1, 2),
            (-2, 1),
            (-2, -1),
            (-1, -2),
            (1, -2),
            (2, -1),   
        )

    def encode(self, move):
        """
        Encodes the given move as knight move, if possible, else returns None
        """
        
        from_rank, from_file, to_rank, to_file = utils.unpack(move)
        delta = (to_rank - from_rank, to_file - from_file)
        is_knight_move = delta in self._DIRECTIONS
        if not is_knight_move:
            return None
        
        knight_move_type = self._DIRECTIONS.index(delta)
        move_type = self._TYPE_OFFSET + knight_move_type

        action = np.ravel_multi_index(
            multi_index = ((move_type, from_rank, to_file)),
            dims = (73, 8, 8)
        )

        return action

    def decode(self, index):
        """
        Decodes the given action as a knight move, if possible.
        """

        move_type, from_rank, from_file = np.unravel_index(index, (73, 8, 8))
        
        is_knight_move = self._TYPE_OFFSET <= move_type and move_type < self._TYPE_OFFSET + self._NUM_TYPES
        if not is_knight_move:
            return None
        
        knight_move_type = move_type - self._TYPE_OFFSET

        delta_rank, delta_file = self._DIRECTIONS[knight_move_type]

        to_rank = from_rank + delta_rank
        to_file = from_file + delta_file

        move = utils.pack(from_rank, from_file, to_rank, to_file)
        return move

class UnderPromotionsEncoding:
    def __init__(self):
        self._TYPE_OFFSET = 64
        self._NUM_TYPES = 9 # 3 directions * 3 piece types
        self._DIRECTIONS = utils.IndexedTuple(-1, 0, 1)
        self._PROMOTIONS = [
            chess.KNIGHT,
            chess.BISHOP,
            chess.ROOK
        ]

    def encode(self, move):
        """
        Encodes the given underpromotion move, if possible, else return None
        """

        from_rank, from_file, to_rank, to_file = utils.unpack(move)
        is_underpromotion = (move.promotion in self._PROMOTIONS and from_rank == 6 and to_rank == 7)
        if not is_underpromotion:
            return None
        
        delta_file = to_file - from_file

        direction_idx = self._DIRECTIONS.index(delta_file)
        promotion_idx = self._PROMOTIONS.index(move.promotion)

        underpromotion_type = np.ravel_multi_index(
            multi_index = ([direction_idx, promotion_idx]),
            dims = (3, 3)
        )

        move_type = self._TYPE_OFFSET + underpromotion_type 

        action = np.ravel_multi_index(
            multi_index = ((move_type, from_rank, from_file)),
            dims = (73, 8, 8)
        )
        
        return action

    def decode(self, index):
        """
        Decodes the given action index into a knight move, if possible, else returns None
        """

        move_type, from_rank, from_file = np.unravel_index(index, (73, 8, 8))

        is_underpromotion = (
        self._TYPE_OFFSET <= move_type
        and move_type < self._TYPE_OFFSET + self._NUM_TYPES
        )

        if not is_underpromotion:
            return None

        underpromotion_type = move_type - self._TYPE_OFFSET

        direction_idx, promotion_idx = np.unravel_index(
            indices=underpromotion_type,
            shape=(3,3)
        )

        direction = self._DIRECTIONS[direction_idx]
        promotion = self._PROMOTIONS[promotion_idx]

        to_rank = from_rank + 1
        to_file = from_file + direction

        move = utils.pack(from_rank, from_file, to_rank, to_file)
        move.promotion = promotion

        return move