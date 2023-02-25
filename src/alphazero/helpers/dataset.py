"""
Encapsulates PyTorch dataset used for training
"""

import chess
import chess.pgn
import os
import torch
import numpy as np

from torch.utils.data import Dataset

# TODO: implement board encoding / decoding

class ChessDataset(Dataset):
    def __init__(self, pgn_path):
        self.games = []
        for filename in os.listdir(pgn_path):
            with open(os.path.join(pgn_path, filename)) as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    self.games.append(game)
                    print(self.__getitem__(0))
                    exit()

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        return NotImplemented

if __name__ == "__main__":
    import chess
    board = chess.Board()

    # TODO: test