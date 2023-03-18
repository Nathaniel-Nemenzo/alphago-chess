import torch
import chess

from game.chess.chess_move_translator import ChessMoveTranslator, QueenMovesTranslator, KnightMovesTranslator, UnderpromotionsTranslator

device = torch.device("cpu")

# TODO: make sure to test different orientations

# test queen move translation
def test_queen_move_translation():
    translator = QueenMovesTranslator()
    move = chess.Move.from_uci("e2e4")
    board = chess.Board()
    
    # check that encoding an action and then decoding the action gives the original action
    idx = translator.encode(move, board)
    reconstructed_move = translator.decode(idx, board)
    
    assert(move == reconstructed_move)

# test knight move translation
def test_knight_move_translation():
    translator = KnightMovesTranslator()
    move = chess.Move.from_uci("g1f3")
    board = chess.Board()

    idx = translator.encode(move, board)
    reconstructed_move = translator.decode(idx, board)

    assert(move == reconstructed_move)

# test underpromotion translation
def test_underpromotion_translation():
    translator = UnderpromotionsTranslator()
    move = chess.Move.from_uci("a7a8")
    move.promotion = chess.BISHOP
    board = chess.Board()

    # test encoding
    idx = translator.encode(move, board)
    reconstructed_move = translator.decode(idx, board)
    assert(move == reconstructed_move)

# test full translationx
def test_full_translation():
    translator = ChessMoveTranslator(device)
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")

    # test difference between black / white encoding
    
    # encode a queen move
    idx = translator.encode(move, board)
    assert(translator.decode(idx, board) == move)

    # encode a knight move
    move = chess.Move.from_uci("g1f3")
    idx = translator.encode(move, board)
    assert(translator.decode(idx, board) == move)

    # encode a underpromotion
    move = chess.Move.from_uci("a7a8")
    move.promotion = chess.BISHOP
    idx = translator.encode(move, board)
    assert(translator.decode(idx, board) == move)

    # encode a queen promotion
    move = chess.Move.from_uci("a7a8")
    move.promotion = chess.QUEEN
    idx = translator.encode(move, board)
    assert(translator.decode(idx, board) == move)