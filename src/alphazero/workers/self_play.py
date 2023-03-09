"""
Encapsulates worker to generate training examples from self-play using neural network-aided Monte Carlo tree search.
"""

import copy
import torch

from alphazero.game import game
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
                 game: game.Game, 
                 model: torch.nn.Module, 
                 args):
        """
        Initialize this worker.

        Args:
            game: Chess game object
            model: PyTorch neural network
            args: 
                - iterations
                - episodes
                - mcts_simulations
                - temp_threshold
        """

        self.game = game
        self.model = model
        self.mcts = MonteCarloTreeSearch()
        
        # Store the current model
        self.old_model = copy.deepcopy(model)
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

        for i in range(self.args.iterations):
            for e in range(self.args.episodes):
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
            
            # Get the temperature (based on the move count)
            temp = int(step < self.args.temp_threshold)

            # Put first training example in examples (the reward can not yet be determined because the game is not finished.)
            improved_policy = self.mcts.getActionProb(board, temp = temp)

            # Append the improved policy to our training examples
            examples.append([])

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