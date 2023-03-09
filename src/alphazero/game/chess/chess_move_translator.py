import chess
import numpy as np

from .utils import rotate, IndexedTuple, unpack, pack
from game.move_translator import MoveTranslator

class ChessMoveTranslator(MoveTranslator):
    """Implements the move encoding from [Silver et al., 2017]
    Based on: https://github.com/iamlucaswolf/gym-chess/tree/master/gym_chess/alphazero/move_encoding
    
    Moves are encoded as indices into a flattened (73, 8, 8) tensor, where each index encodes a possible move.
    The first two dimensions correspond to the square from which the piece is picked up. The last dimension denotes the 
    "move type", which describes how the selected piece is moved from its current position.
    Silver et al. defines three move types:

        - queen moves: move the pieces horizontally, vertically, or diagonally for any number of squares (56 total moves)
        - knight moves: move the piece in an L shape (two squares either horizontally or vertically, followed by one square orthogonally) (8 total moves)
        - underpromotions: let a pawn move from the 7th to the 8th rank and promote to either a knight, bishop, or rook (hence 'under').
                           Moving a pawn to the 8th rank with a queen move is automatically assumed to be a queen promotion. (9 total moves)

    Args:
        MoveTranslator (_type_): _description_
    """
    def __init__(self):
        self.queenMovesTranslator = QueenMovesTranslator()
        self.knightMovesTranslator = KnightMovesTranslator()
        self.underpromotionTranslator = UnderpromotionsTranslator()

    def encode(self, move: tuple([chess.Move, chess.Color])) -> int:
        """Encodes a chess.Move instance into a corresponding action integer.

        Args:
            move (tuple): Tuple of the form (move, turn). move is assumed to be in the perspective of white, so if turn is chess.BLACK, move will be rotated to the perspective of black.

        Raises:
            ValueError: Raise a value error if a move cannot be encoded.

        Returns:
            int: Returns a int describing an index in a flattened (73, 8, 8) tensor, as above. Notice that the mapping from move to integer is not the same for different orientations.
        """
        turn = move[1]
        move = move[0]
        if turn == chess.BLACK:
            move = rotate(move)

        action = self.queenMovesTranslator.encode(move)
        
        if action is None:
            action = self.knightMovesTranslator.encode(move)

        if action is None:
            action = self.underpromotionTranslator.encode(move)

        # Invalid move
        if action is None:
            raise ValueError(f"{move} is not considered to be a valid move.")

        return action

    def decode(self, move: tuple([int, chess.Color, bool])) -> chess.Move:
        """Converts an encoded action (as an integer) into its corresponding action.

        Args:
            move (tuple): Tuple of the form (action as integer, turn, is_pawn). Action is assumed to be in the perspective of white, so if turn is chess.BLACK, the resulting move will be in the perspective of player black.

        Raises:
            ValueError: If the action cannot be decoded

        Returns:
            chess.Move: Move in the given orientation corresponding to the given action integer.
        """
        pawn = move[2]
        turn = move[1]
        action = move[0]
        move = self.queenMovesTranslator.decode(action)

        is_queen_move = move is not None
        
        # Sequentially check all encodings
        if not move:
            move = self.knightMovesTranslator.decode(action)

        if not move:
            move = self.underpromotionTranslator.decode((action, turn))

        if not move:
            raise ValueError(f"{move} is not a valid action.")

        # Reorient action if player black
        if turn == chess.BLACK:
            move = rotate(move)

        # Moving a pawn to the opponent's home rank with a queen move is automatically assumed to be a queen underpromotion.
        # Add this situation manually because the QueenMoves class has no access to board state or piece type
        if is_queen_move:
            to_rank = chess.square_rank(move.to_square)
            is_promoting_move = (
                (to_rank == 7 and turn == chess.WHITE) or 
                (to_rank == 0 and turn == chess.BLACK)
            )

            if pawn and is_promoting_move:
                move.promotion = chess.QUEEN

        return move

class QueenMovesTranslator(MoveTranslator):
    def __init__(self):
        self._TYPE_OFFSET = 0
        self._NUM_TYPES = 56
        self._DIRECTIONS = IndexedTuple(
            (0, 1),  # N
            (1, 1),  # NE
            (1, 0),  # E
            (1, -1),  # SE
            (0, -1),  # S
            (-1, -1),  # SW
            (-1, 0),  # W
            (-1, 1)  # NW
        )

    def encode(self, move: chess.Move) -> int:
        """
        Encode a queen move into action index representation, if possible, else returns None
        """
        from_rank, from_file, to_rank, to_file = unpack(move)
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

    def decode(self, move: int) -> chess.Move:
        """
        Decode a moFve in action index representation to a queen move
        """
        move_type, from_rank, from_file = np.unravel_index(move, (73, 8, 8))
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

        move = pack(from_rank, from_file, to_rank, to_file)
        return move

class KnightMovesTranslator(MoveTranslator):
    def __init__(self):
        self._TYPE_OFFSET = 56
        self._NUM_TYPES = 8
        self._DIRECTIONS = IndexedTuple(
            (2, 1),
            (1, 2),
            (-1, 2),
            (-2, 1),
            (-2, -1),
            (-1, -2),
            (1, -2),
            (2, -1),   
        )

    def encode(self, move: chess.Move) -> int:
        """
        Encodes the given move as knight move, if possible, else returns None
        """
        
        from_rank, from_file, to_rank, to_file = unpack(move)
        delta = (to_rank - from_rank, to_file - from_file)
        is_knight_move = delta in self._DIRECTIONS
        if not is_knight_move:
            return None
        
        knight_move_type = self._DIRECTIONS.index(delta)
        move_type = self._TYPE_OFFSET + knight_move_type

        action = np.ravel_multi_index(
            multi_index = ((move_type, from_rank, from_file)),
            dims = (73, 8, 8)
        )

        return action

    def decode(self, move: int) -> chess.Move:
        """
        Decodes the given action as a knight move, if possible.
        """

        move_type, from_rank, from_file = np.unravel_index(move, (73, 8, 8))
        
        is_knight_move = self._TYPE_OFFSET <= move_type and move_type < self._TYPE_OFFSET + self._NUM_TYPES
        if not is_knight_move:
            return None
        
        knight_move_type = move_type - self._TYPE_OFFSET

        delta_rank, delta_file = self._DIRECTIONS[knight_move_type]

        to_rank = from_rank + delta_rank
        to_file = from_file + delta_file

        move = pack(from_rank, from_file, to_rank, to_file)
        return move

class UnderpromotionsTranslator(MoveTranslator):
    def __init__(self):
        self._TYPE_OFFSET = 64
        self._NUM_TYPES = 9 # 3 directions * 3 piece types
        self._DIRECTIONS = IndexedTuple(-1, 0, 1)
        self._PROMOTIONS = [
            chess.KNIGHT,
            chess.BISHOP,
            chess.ROOK
        ]

    def encode(self, move: chess.Move) -> int:
        from_rank, from_file, to_rank, to_file = unpack(move)
        is_underpromotion = (move.promotion in self._PROMOTIONS and (from_rank == 6 and to_rank == 7) or (from_rank == 1 and to_rank == 0))
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

    def decode(self, move: tuple([int, chess.Color])) -> chess.Move:
        move_type, from_rank, from_file = np.unravel_index(move[0], (73, 8, 8))
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
        # print(direction)
        promotion = self._PROMOTIONS[promotion_idx]

        to_rank = from_rank + 1 if move[1] == chess.WHITE else from_rank - 1
        to_file = from_file + direction
        print(to_rank)

        ret = pack(from_rank, from_file, to_rank, to_file)
        ret.promotion = promotion

        return ret