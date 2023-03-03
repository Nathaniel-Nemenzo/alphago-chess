import torch
import chess
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

    # Make 8 moves to build up history
    board.push(chess.Move.from_uci("a2a3"))
    translator._ChessBoardTranslator__push_into_history(board, tensor, 7)

    board.push(chess.Move.from_uci("a7a6"))
    translator._ChessBoardTranslator__push_into_history(board, tensor, 6)

    board.push(chess.Move.from_uci("b2b3"))
    translator._ChessBoardTranslator__push_into_history(board, tensor, 5)

    board.push(chess.Move.from_uci("b7b6"))
    translator._ChessBoardTranslator__push_into_history(board, tensor, 4)

    board.push(chess.Move.from_uci("c2c3"))
    translator._ChessBoardTranslator__push_into_history(board, tensor, 3)

    board.push(chess.Move.from_uci("c7c6"))
    translator._ChessBoardTranslator__push_into_history(board, tensor, 2)

    board.push(chess.Move.from_uci("d2d3"))
    translator._ChessBoardTranslator__push_into_history(board, tensor, 1)

    board.push(chess.Move.from_uci("d7d6"))
    translator._ChessBoardTranslator__push_into_history(board, tensor, 0)

    # Get view of board from white side
    view = translator._ChessBoardTranslator__view(tensor, chess.WHITE)

    # Check all time steps
    for i in range(8):
        assert_board_state_post_view(view, i)

    # Get view of board from black side
    view = translator._ChessBoardTranslator__view(tensor, chess.BLACK)
    # print(view[0])
    # print(view[1])
    # print(view[2])
    # print(view[3])
    # print(view[4])
    # print(view[5])



    for i in range(8):
        assert_board_state_post_view(view, i)


def test_meta():
    pass

def test_encode():
    pass
    # board = chess.Board()

    # # Make 8 moves to build up history
    # board.push(chess.Move.from_uci("a2a3"))
    # board.push(chess.Move.from_uci("a7a6"))
    # board.push(chess.Move.from_uci("b2b3"))
    # board.push(chess.Move.from_uci("b7b6"))
    # board.push(chess.Move.from_uci("c2c3"))
    # board.push(chess.Move.from_uci("c7c6"))
    # board.push(chess.Move.from_uci("d2d3"))
    # board.push(chess.Move.from_uci("d7d6"))

    # # Encode
    # tensor = translator.encode(board)

    # # Check all time steps
    # for i in range(8):
    #     assert_board_state_post_view(tensor, i)