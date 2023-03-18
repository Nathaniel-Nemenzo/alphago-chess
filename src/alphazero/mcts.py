"""
Encapsulates algorithm for Neural network-aided Monte Carlo Tree Search as proposed by [Silver et al., 2017]

Based on: https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py

# TODO: vectorize
"""

import chess
import torch
import torch.nn as nn

from math import sqrt
from alphazero.game.game import Game

from alphazero.game.board_translator import BoardTranslator
from alphazero.game.move_translator import MoveTranslator

class MonteCarloTreeSearch:
    """
    Handles Monte Carlo tree search
    """
    def __init__(self, 
                 device: torch.device,
                 model: nn.Module, 
                 game: Game,
                 board_translator: BoardTranslator,
                 move_translator: MoveTranslator,
                 args):
        self.device = device
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

        self.game = game
        self.board_translator = board_translator
        self.move_translator = move_translator

    def search(self, board: chess.Board):
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
        s = self.game.stringRepresentation(board)

        # Get all the legal moves for the current player
        legal_moves = torch.Tensor(self.game.getValidActions(board), device = self.device)

        # Check if the game is over and propagate the values upward
        if self.game.getGameEnded(board):
            return -self.game.getRewards(board)

        # Check if this is a leaf node (not visited yet)
        if s not in self.P_s:
            p, v = self.model.forward(self.board_translator.encode(board))

            # Mask invalid values using a mask tensor
            mask = torch.zeros(p.shape, device = self.device)
            mask[0, legal_moves] = 1
            p = p * mask

            # Renormalize policy
            p_sum = torch.sum(p)
            if p_sum <= 0:
                p += mask
            
            p /= torch.sum(p)

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

        # (s, best_a) represents the best action to take from the current node with the current thread.
        # We must use virtual loss to lower the probability of other threads taking this (s, a) pair.
        # self.Q_sa[(s, best_a)] -= self.args.virtual_loss

        # Decode the action to game representation
        best_a = self.move_translator.decode(best_a)

        # Recurse on the resulting state
        next_s = self.game.getNextState(board, best_a)

        # Get the value of the action from this state
        v = self.search(next_s)

        # Before updating Q values, remove the virtual loss from this action.
        # self.Q_sa[(s, best_a)] += self.args.virtual_loss

        # Update Q values and visit counts for the sampled action
        self.Q_sa[(s, best_a)] = (self.N_sa[(s, best_a)] * self.Q_sa[(s, best_a)] + v) / (self.N_sa[(s, best_a)] + 1)
        self.N_sa[(s, best_a)] += 1

        # Update the number of times we have visited this state
        self.N_s[s] += 1

        return -v
    
    def improvedPolicy(self, board: chess.Board, temp = 1):
        """
        Performs num_mcts_simulations simulations of MCTS starting from the given state.

        Returns:
            probs: a policy tensor of  where the probability of the ith action is proportional to N_sa[(s, a)] ** (1./temp)
        """
        
        # Perform simulations
        for _ in range(self.args.num_mcts_simulations):
            self.search(board)

        s = board.fen()

        # Get (s, a) counts
        counts = torch.Tensor([self.N_sa.get((s, a), 0) for a in range(self.game.getActionSize())], device = self.device)

        # If the temperature == 0, pick the move with the maximum counts
        if temp == 0:
            max_index = torch.argmax(counts)
            result = torch.zeros((1, len(counts)))
            result[0][max_index] = 1
            return result
        
        # If the temperature != 0, apply the temperature over all counts
        probs = counts ** (1. / temp)

        # Normalize the probabilities
        probs = torch.softmax(probs)
        return probs
        
            