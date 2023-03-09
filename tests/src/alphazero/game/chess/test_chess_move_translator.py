import torch
import chess
import numpy as np

from game.chess.chess_move_translator import ChessMoveTranslator, QueenMovesTranslator, KnightMovesTranslator, UnderpromotionsTranslator

# TODO: make sure to test different orientations

# test queen move translation
def test_queen_move_translation():
    translator = QueenMovesTranslator()
    move = chess.Move.from_uci("e2e4")
    
    # check that encoding an action and then decoding the action gives the original action
    idx = translator.encode(move)
    reconstructed_move = translator.decode(idx)
    
    assert(move == reconstructed_move)

# test knight move translation
def test_knight_move_translation():
    translator = KnightMovesTranslator()
    move = chess.Move.from_uci("g1f3")

    idx = translator.encode(move)
    reconstructed_move = translator.decode(idx)

    assert(move == reconstructed_move)

# test underpromotion translation
def test_underpromotion_translation():
    translator = UnderpromotionsTranslator()
    move = chess.Move.from_uci("a7a8")
    move.promotion = chess.BISHOP

    # test white encoding
    idx = translator.encode(move)
    reconstructed_move = translator.decode((idx, chess.WHITE))
    assert(move == reconstructed_move)

    # can't test black encoding in this test, because moves aren't rotated to the correct orientation in this class, but rather in the full class 
    # reconstructed_move = translator.decode((idx, chess.BLACK))
    # assert(move == reconstructed_move)

# test full translationx
def test_full_translation():
    translator = ChessMoveTranslator()
    move = chess.Move.from_uci("e2e4")

    # test difference between black / white encoding
    idx_white = translator.encode((move, chess.WHITE))
    idx_black = translator.encode((move, chess.BLACK))
    assert(idx_white != idx_black)
    
    # encode a queen move
    # make sure that the indices encoded with their respective turns get the same move
    assert(translator.decode((idx_white, chess.WHITE, False)) == move)
    assert(translator.decode((idx_black, chess.BLACK, False)) == move)

    # encode a knight move
    move = chess.Move.from_uci("g1f3")
    idx_white = translator.encode((move, chess.WHITE))
    idx_black = translator.encode((move, chess.BLACK))
    assert(translator.decode((idx_white, chess.WHITE, False)) == move)
    assert(translator.decode((idx_black, chess.BLACK, False)) == move)

    # encode a underpromotion
    move = chess.Move.from_uci("a7a8")
    move.promotion = chess.BISHOP
    idx_white = translator.encode((move, chess.WHITE))
    idx_black = translator.encode((move, chess.BLACK))
    assert(translator.decode((idx_white, chess.WHITE, False)) == move)
    assert(translator.decode((idx_black, chess.BLACK, False)) == move)