"""
Encapsulates algorithm for Neural network-aided Monte Carlo Tree Search as proposed by [Silver et al., 2017]
"""

import chess
import torch

from math import sqrt

from helpers.move import MoveRepresentation
from helpers.board import BoardRepresentation

move_encoder = MoveRepresentation()
board_encoder = BoardRepresentation()

class MCTS:
    """
    Handles Monte Carlo tree search
    """
    def __init__(self, model, args):
        self.model = model
        self.args = args

        # Store Q values for (s, a)
        self.Q_sa = {}

        # Store number of times edge (s, a) has been taken
        self.N_sa = {}

        # Store number of times state s has been visited
        self.N_s = {}

        # Store the initial policy predicted by the model for each state
        # This is normalized over the valid actions
        self.P_s = {}

    def search(self, board, model):
        """
        Performs one iteration of MCTS. It is recursively called until a leaf node is found. The algorithm is as follows:

        0. Initialize an empty search tree with s (board) as the root.
        1. Compute the action a that maximizes u(s, a), which denotes the upper confidence bound on Q-values. From the paper, the action chosen has the maximum u(s, a) over all actions.
            a. If state s' (by doing a) exists in the tree, recurse on s'
            b. If s' does not exist, add the new state to the tree and initialize P(s', .) ~ P_theta(s') and v(s') ~ V_theta(s'). Then, initialize Q(s', a) and N(s', a) to 0 for all a
            c. If we encounter a terminal state, propagate the actual reward
        2. Propagate values upwards and update N_s, N_sa, and Q_sa
        """

        # Get unique identifier for game state
        s = board.fen()

        # Get all the legal moves for the current player
        legal_moves = torch.Tensor([move_encoder.encode(move) for move in board.legal_moves if board.color_at(move.from_square) == board.turn])

        # Check if the game is over and propagate the values upward
        if board.is_game_over():
            if board.result() == "1-0":
                # White won
                return 1 if board.turn == chess.BLACK else -1
            elif board.result() == "0-1":
                # Black won
                return 1 if board.turn == chess.WHITE else -1
            else:
                return 0

        # Check if this is a leaf node (not visited yet)
        if s not in self.P_s:
            p, v = model.forward(board_encoder.observation(board))

            # Mask invalid values using a mask tensor
            mask = torch.zeros(p.shape)
            mask[0, legal_moves] = 1
            p = p * mask

            # Renormalize policy
            p_sum = torch.sum(p)
            if p_sum <= 0:
                p += mask
            p /= p_sum

            self.P_s[s] = p
            self.N_s[s] = 0
            return -v

        # Initialize Q values and visit counts for (s, a)
        for a in legal_moves:
            if (s, a) not in self.Q_sa:
                self.Q_sa[(s, a)] = 0
                self.N_sa[(s, a)] = 0

        cur_best_u = float('-inf')
        best_a = -1

        # Choose the action with the highest u(s, a) value
        for a in legal_moves:
            u = self.Q_sa[(s, a)] + self.args.cpuct * self.P_s[s][a] * (sqrt(self.N_s[s]) / (1 + self.N_sa[(s, a)]))
            if u > cur_best_u:
                cur_best_u = u
                best_a = a

        # Recurse on the resulting state
        chess_move = move_encoder.decode(best_a)
        board.push(chess_move)
        v = self.search(board)
        board.pop()

        # Update Q values and visit counts for the sampled action
        self.Q_sa[(s, best_a)] = (self.N_sa[(s, a)] * self.Q_sa[(s, a)] * v) / (self.N_sa[(s, a)] + 1)
        self.N_sa[(s, a)] += 1

        # Update the number of times we have visited this state
        self.N_s[s] += 1

        return -v
            