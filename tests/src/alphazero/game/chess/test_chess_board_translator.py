import torch
import chess
import random
from game.chess.chess_board_translator import ChessBoardTranslator

translator = ChessBoardTranslator()

def assert_board_state_pre_view(tensor, i):
    # Pawns
    assert(torch.sum(tensor[i][0]) == 8)
    assert(torch.sum(tensor[i][6]) == 8)

    # Knight
    assert(torch.sum(tensor[i][1]) == 2)
    assert(torch.sum(tensor[i][7]) == 2)

    # Bishop
    assert(torch.sum(tensor[i][2]) == 2)
    assert(torch.sum(tensor[i][8]) == 2)

    # Rook
    assert(torch.sum(tensor[i][3]) == 2)
    assert(torch.sum(tensor[i][9]) == 2)

    # King
    assert(torch.sum(tensor[i][4]) == 1)
    assert(torch.sum(tensor[i][10]) == 1)

    # Queen
    assert(torch.sum(tensor[i][5]) == 1)
    assert(torch.sum(tensor[i][11]) == 1)

def assert_board_state_post_view(tensor, i):
    # Pawns
    assert(torch.sum(tensor[14 * i]) == 8)
    assert(torch.sum(tensor[14 * i + 6]) == 8)

    # Knight
    assert(torch.sum(tensor[14 * i + 1]) == 2)
    assert(torch.sum(tensor[14 * i + 7]) == 2)

    # Bishop
    assert(torch.sum(tensor[14 * i + 2]) == 2)
    assert(torch.sum(tensor[14 * i + 8]) == 2)

    # Rook
    assert(torch.sum(tensor[14 * i + 3]) == 2)
    assert(torch.sum(tensor[14 * i + 9]) == 2)

    # King
    assert(torch.sum(tensor[14 * i + 4]) == 1)
    assert(torch.sum(tensor[14 * i + 10]) == 1)

    # Queen
    assert(torch.sum(tensor[14 * i + 5]) == 1)
    assert(torch.sum(tensor[14 * i + 11]) == 1)

def test_push_into_history():
    tensor = torch.zeros((8, 14, 8, 8))
    board = chess.Board() # default state

    tensor = translator._ChessBoardTranslator__push_into_history(board, tensor, 0)

    assert_board_state_pre_view(tensor, 0)

    board.push(chess.Move.from_uci("g1f3"))
    tensor = translator._ChessBoardTranslator__push_into_history(board, tensor, 1)
    assert_board_state_pre_view(tensor, 1)

def test_view():
    tensor = torch.zeros(8, 14, 8, 8)
    board = chess.Board()

    # Make 8 moves to build up history (no captures)
    for i in range(8):
        legal_moves = list(filter(lambda move: not board.is_capture(move), board.legal_moves)) # get all non-capture legal moves
        if not legal_moves: # if no legal moves left, break out of the loop
            break
        move = random.choice(legal_moves) # choose a random non-capture legal move
        board.push(move) # make the move on the board
        tensor = translator._ChessBoardTranslator__push_into_history(board, tensor, i)

    # Get view of board from white side
    view = translator._ChessBoardTranslator__view(tensor, chess.WHITE)

    # Check all time steps
    for i in range(8):
        assert_board_state_post_view(view, i)

    # Get view of board from black side
    view = translator._ChessBoardTranslator__view(tensor, chess.BLACK)

    for i in range(8):
        assert_board_state_post_view(view, i)


def test_encode():
    board = chess.Board()

    # Make 8 moves to build up history
    for i in range(9):
        legal_moves = list(filter(lambda move: not board.is_capture(move), board.legal_moves)) # get all non-capture legal moves
        if not legal_moves: # if no legal moves left, break out of the loop
            break
        move = random.choice(legal_moves) # choose a random non-capture legal move
        board.push(move) # make the move on the board


    # Encode
    tensor = translator.encode(board)

    assert(tensor.shape == (119, 8, 8))

    # Check all time steps
    for i in range(8):
        assert_board_state_post_view(tensor, i)

    assert(torch.sum(tensor[112]) == 0)
    assert(torch.sum(tensor[113]) == 5 * 64)
    assert(torch.sum(tensor[114]) == 64)
