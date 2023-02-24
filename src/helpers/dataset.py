import chess
import chess.pgn
import os
import torch
import numpy as np

from torch.utils.data import DataSet, DataLoader

class ChessDataset(DataSet):
    def __init__(self, pgn_path):
        self.games = []
        for filename in os.listdir(pgn_path):
            with open(os.path.join(pgn_path, filename)) as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    self.games.append(game)

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        game = self.games[idx]
        board = game.board()
        X = []
        Y = []
        for move in game.mainline_moves():
            X.append(board_to_tensor(move))
            Y.append(move_to_index(move))
            board.push(move)
        return torch.stack(X), torch.LongTensor(Y)

def board_to_tensor(board):
    pieces = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']

    # Initialize tensor representation
    tensor = np.zeros((8, 8, 14))

    # Loop through squares and convert to tensor
    for r in range(8):
        for c in range(8):
            piece = board.piece_at(chess.square(c, r))

            if piece is not None:
                piece_idx = pieces.index(piece.symbol())
                color = int(piece.color)

                # Set tensor value (map from 0-13)
                tensor[r, c, color * 6 + piece_idx] = 1

def move_to_index(move):
    pass