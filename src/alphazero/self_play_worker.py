"""
Encapsulates worker to generate training examples from self-play using neural network-aided Monte Carlo tree search.

Based on: https://github.com/suragnair/alpha-zero-general/blob/master/Coach.py
"""

import os
import torch
import tqdm
import queue

from alphazero.replay_buffer import ReplayBuffer
from alphazero.game import game
from game.board_translator import BoardTranslator
from game.move_translator import MoveTranslator
from mcts import MonteCarloTreeSearch

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
                 accepted_model_queue: queue.Queue,
                 training_model_queue: queue.Queue,
                 replay_buffer: ReplayBuffer,
                 board_translator: BoardTranslator,
                 move_translator: MoveTranslator,
                 args: dict):
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
        self.accepted_model_queue = accepted_model_queue
        self.training_model_queue = training_model_queue

        self.mcts = None
        self.replay_buffer = replay_buffer
        self.board_translator = board_translator
        self.move_translator = move_translator
        self.args = args

    def start(self):
        """Perform num_iterations iterations with num_eps episodes of self-play in each iteration. Training examples with the improved policy from MCTS are generated here.
        """
        while True:
            # Check for the latest model in the accepted queue
            try:
                latest_model = self.accepted_model_queue.get(timeout = self.args.accepted_model_queue_timeout)
            except queue.Empty:
                continue

            # Set the model to evaluation model
            latest_model.eval()

            # Perform num_iters iterations
            for i in range(self.args.num_self_play_iterations):

                # Keep track of the training examples for each iteration
                iteration_examples = []

                # Training episodes
                for _ in tqdm(range(self.args.num_eps), desc = "self play"):
                    # Reset the search tree for each episode
                    self.mcts = MonteCarloTreeSearch(latest_model, self.game, self.board_translator, self.move_translator, self.args)

                    # Add the training examples from each episode to the replay buffer
                    examples = self.episode()
                    self.replay_buffer.extend(examples)

                    # Add the training examples from this episode to the iteration examples
                    iteration_examples.append(examples)

                # Put the model in the training model queue to be trained
                self.training_model_queue.put(latest_model)

                # Save the iteration examples
                save_training_examples(iteration_examples, i, self.args.training_example_path)
    
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
            improved_policy = self.mcts.improvedPolicy(board, temp = temp)

            # Append the improved policy to our training examples
            examples.append([self.game.getBoard(board), improved_policy.reshape(1, -1), None])

            # Now, sample an action from the improved policy
            # Switch to Tensor implementation
            a = torch.multinomial(improved_policy, num_samples = 1, replacement = False)

            # Get the next state with the action
            board = self.game.nextState(board, a)

            # Determine whether the game has ended 
            r = self.game.gameRewards(board)

            # Put the reward in the training examples
            if r != None:
                return [(x[0], x[1], r * ((-1) ** (x[0].turn != board.turn))) for x in examples]
            
def save_training_examples(iteration_examples, iteration, path):
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create the '../../data' directory if it doesn't exist
    data_dir = os.path.join(current_dir, '..', '..', path)
    os.makedirs(data_dir, exist_ok=True)

    # Construct the filename
    filename = f'iteration_examples_{iteration}.pt'

    # Save the training examples as a PyTorch file
    torch.save(iteration_examples, os.path.join(data_dir, filename))

def load_training_examples(self, iteration, path):
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the data directory path
    data_dir = os.path.join(current_dir, '..', '..', path)

    # Construct the filename
    filename = f'iteration_examples_{iteration}.pt'

    # Load the training examples from the PyTorch file
    iteration_examples = torch.load(os.path.join(data_dir, filename))

    return iteration_examples