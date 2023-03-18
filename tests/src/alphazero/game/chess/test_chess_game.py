import torch
import chess

from game.chess.chess_game import ChessGame
from game.chess.chess_board_translator import ChessBoardTranslator
from game.chess.chess_move_translator import ChessMoveTranslator

device = torch.device("cpu")
board_translator = ChessBoardTranslator(device)
move_translator = ChessMoveTranslator(device)

# Chess game is stateless
chess_game = ChessGame(device, board_translator, move_translator)

def test_getInitBoard():
    # init board
    init = chess.Board()

    assert(init == chess_game.getInitBoard())

def test_getBoardSize():
    assert(chess_game.getBoardSize() == (8, 8))

def test_getActionSize():
    assert(chess_game.getActionSize() == 4672)

def test_getNextState():
    # testing board
    board = chess.Board()

    # copy board
    copy_board = chess.Board()
    move = chess.Move.from_uci("e2e3")

    # encode move before pushing
    move_ind = move_translator.encode(move, copy_board)

    # test get next state
    copy_board = chess_game.getNextState(copy_board, move_ind)

    # push move on board
    board.push(move)

    assert(board == copy_board)

def test_getValidMoves():
    # initialize default chess board
    board = chess.Board()

    # current turn is white, get valid moves
    moves = chess_game.getValidMoves(board)
    actions = [move_translator.encode(move, board) for move in moves]

    # test all moves
    for action in actions:
        move = move_translator.decode(action, board)

        # make sure move is legal
        assert(board.is_legal(move))

    # make a move
    move_made = move_translator.decode(actions[0], board)
    board.push(move_made)

    # now, test all moves (black side)
    black_moves = chess_game.getValidMoves(board)
    black_actions = [move_translator.encode(move, board) for move in black_moves]

    # test all moves
    for action in black_actions:
        move = move_translator.decode(action, board)

        # make sure move is legal
        assert(board.is_legal(move))

def test_getGameEnded():
    board = chess.Board('4k3/4Q3/4K3/8/8/8/8/8 b - - 0 1')
    assert(chess_game.getGameEnded(board) == True)

def test_getRewards():
    # white wins
    white_wins = chess.Board('4k3/4Q3/4K3/8/8/8/8/8 b - - 0 1')

    # black wins
    black_wins = chess.Board('8/8/8/8/8/3k4/3q4/3K4 w - - 0 1')

    # check
    # white_wins.turn = chess.WHITE
    # these should both be -1, because when each color wins, the OTHER color is the current player, and so this is a loss from the OTHER PLAYER's view. 
    assert(chess_game.getRewards(white_wins) == -1)
    assert(chess_game.getRewards(black_wins) == -1)

    # draw
    tie = chess.Board('8/8/8/8/8/5k2/8/5K2 w - - 0 1')
    assert(chess_game.getRewards(tie) == 0)

    # not done
    board = chess.Board()
    assert(chess_game.getRewards(board) is None)
