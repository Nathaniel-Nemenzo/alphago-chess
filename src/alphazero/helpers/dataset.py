"""
Encapsulates PyTorch dataset used for training
"""

import chess
import chess.pgn
import os
import torch
import numpy as np

from torch.utils.data import Dataset

class ChessDataset(Dataset):
    """
    Dataset storing training examples. A single training example is of the form (BoardRepresentation, MoveRepresentation), where
    BoardRepresentation represents a (8, 8, 119) encoding of a board (see board.py and [Silver et al. 2017]), and a MoveRepresentation
    represents a scalar action index (see move.py and [Silver et al. 2017]).

    Args:
        - pgn_path: Path of where to find the .pgn files, which hold training examples.
        - num_examples: Number of training examples to include in the dataset. 
            Default: 100
            Pass in None for all examples in the path
    """
    def __init__(self, pgn_path, num_examples):
        self.games = []
        self.board_representations = []
        games = 0

        # Read games into buffer
        for filename in os.listdir(pgn_path):
            with open(os.path.join(pgn_path, filename)) as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    self.games.append(game)
                    games += 1
                    if num_examples not None and games == num_examples:
                        break

        # Convert games into board representations

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        """
        Get an individual training example from the dataset.
        Return value is a tuple of ()
        """
        # Get the game
        game = self.games[idx]

        # 


if __name__ == "__main__":
    import chess
    board = chess.Board()

    # TODO: test