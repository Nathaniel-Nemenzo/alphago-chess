"""
Encapsulates worker to generate training examples from self-play using neural network-aided Monte Carlo tree search.
"""

import os
import sys
import torch
import chess

from game.chess import ChessGame
from agent.model import AlphaZeroNetwork
from mcts.mcts import MonteCarloTreeSearch

def start():
    SelfPlayWorker.start()

class SelfPlayWorker:
    """
    Perform policy iteration through self-play. Complete training algorithm is as follows.
    
    0. Initialize the model with random weights, thus starting with a random policy and value network
    1. In each iteration of the algorithm, play a number of games of self-play
        a. In each turn of a game, perform a fixed number of MCTS simulations starting from the current state
        b. Pick a move by sampling from the improved policy, giving us a training example (s_t, pi_t, _)
            - Fill in the reward at the end of the game
            - Preserve the search tree during a game
    2. At the end of the iteration, the model is trained with the training examples.
        a. The old and the new models are pit against each other.
        b. If the new model wins more than a set threshold fraction of games, the model is updated to the new model.

    """
    def __init__(self, 
                 game: ChessGame, 
                 model: torch.nn.Module, 
                 args):
        """
        Initialize this worker.

        Args:
            iterations: Number of policy iterations that we want
            episodes: Number of training episodes (entire game) that we want to collect for each policy iteration
            output_path: Specifies the path that we want to write our training examples to 
            output_path: Specifies the path to the model to begin policy iteration on. If this is None, then use a model with random weights.
            mcts_simulations: Number of iterations to use in Monte Carlo Tree Search (essentially how much of the subtree we simulate), the more iterations the better the policy.
            c_puct: PUCT constant corresponding on how much we want to exploit vs. explore in MCTS.
        """

        self.game = game
        self.model = model
        
        # Store the current model
        self.old_model = AlphaZeroNetwork()
        self.old_model.load_state_dict(self.model.state_dict())

        self.args = args
        self.examples = []
        
        # We want our model and old model in evalulation mode, because we are using model outputs in MCTS
        self.model.eval()
        self.old_model.eval()

    def start(self):
        """
        Begin policy iteration through self play (as described above)
        """

        for i in range(self.iterations):
            for e in range(self.episodes):
                examples += self.episode()
    
    def episode(self):
        """
        Execute one episode of MCTS self-play. Each turn is added as a training example to the stored examples. The game is played
        until end. After the game ends, the outcome is used to assign values to each examples in the stored examples
        """
        examples = []
        board = self.game.newBoard()
        step = 0

        while True:
            step += 1
            for _ in range(self.mcts_simulations):
                # Simulate games
                mcts.search(s, self.model)
            
            # Put first training example in examples (the reward can not yet be determined because the game is not finished.)
            policy = mcts.P_s[s]
            examples.append([s, policy, None])

            # Now, sample an action from the improved policy
            # Switch to Tensor implementation
            a = random.choice(policy.shape[1], p = policy)

            # Get the next state with the action


    def train(self):
        """
        Train the model stored by this class on the training examples stored by this class. 
        """
        return NotImplemented

    def pit(self, updated_model):
        return NotImplemented