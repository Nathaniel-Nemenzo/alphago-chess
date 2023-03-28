import torch
import chess

from game.chess.chess_move_translator import ChessMoveTranslator, QueenMovesTranslator, KnightMovesTranslator, UnderpromotionsTranslator

device = torch.device("cpu")

# TODO: make sure to test different orientations (encoding should be the same whether black or white pieces)

# test queen move translation
def test_queen_move_translation():
    translator = QueenMovesTranslator()
    expected = chess.Move.from_uci("e2e4")
    board = chess.Board()

    idx = translator.encode(expected, board)
    actual = translator.decode(idx, board)
    assert(expected == actual)

# test knight move translation
def test_knight_move_translation():
    translator = KnightMovesTranslator()
    expected = chess.Move.from_uci("g1f3")
    board = chess.Board()

    idx = translator.encode(expected, board)
    actual = translator.decode(idx, board)
    assert(expected == actual)

# test underpromotion translation
def test_underpromotion_translation():
    promotions = ['b','n','r']
    for promotion in promotions:
        translator = UnderpromotionsTranslator()
        expected = chess.Move.from_uci("a7a8" + promotion)
        board = chess.Board()

        idx = translator.encode(expected, board)
        actual = translator.decode(idx, board)
        assert(expected == actual)

# test full translations
# test difference black/white encoding
def test_full_translation():
    translator = ChessMoveTranslator(device)
    white_board = chess.Board()
    black_board = chess.Board()
    black_board.turn = 0

    # encode a queen move
    white_view = chess.Move.from_uci("e2e4")
    white_idx = translator.encode(white_view, white_board)
    assert(translator.decode(white_idx, white_board) == white_view)

    # encode the black perspective
    black_view = chess.Move.from_uci("d7d5")
    black_idx = translator.encode(black_view, black_board)
    assert(white_idx == black_idx)
    assert(translator.decode(black_idx, black_board) == black_view)

    # encode a knight move
    white_view = chess.Move.from_uci("g1f3")
    white_idx = translator.encode(white_view, white_board)
    assert(translator.decode(white_idx, white_board) == white_view)
    
    # encode the black perspective
    black_view = chess.Move.from_uci("b8c6")
    black_idx = translator.encode(black_view, black_board)
    assert(black_idx == white_idx)
    assert(translator.decode(black_idx, black_board) == black_view)

    # encode a underpromotion
    white_view = chess.Move.from_uci("a7a8")
    white_view.promotion = chess.BISHOP
    white_idx = translator.encode(white_view, white_board)
    assert(translator.decode(white_idx, white_board) == white_view)

    # encode the black perspective
    black_view = chess.Move.from_uci("h2h1")
    black_view.promotion = chess.BISHOP
    black_idx = translator.encode(black_view, black_board)
    assert(black_idx == white_idx)
    assert(translator.decode(black_idx, black_board) == black_view)

    # try black perpective
    black_view = chess.Move.from_uci("h7h8")
    black_view.promotion = chess.ROOK
    black_idx = translator.encode(black_view, black_board)

    # encode from white perspective
    white_view = chess.Move.from_uci("a2a1")
    white_view.promotion = chess.ROOK
    white_idx = translator.encode(white_view, white_board)
    assert(black_idx == white_idx)
    assert(translator.decode(black_idx, black_board) == black_view)
    assert(translator.decode(white_idx, white_board) == white_view)
    