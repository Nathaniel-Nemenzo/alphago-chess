"""
Encapsulates worker to generate training examples from self-play using neural network-aided Monte Carlo tree search.

Based on: https://github.com/suragnair/alpha-zero-general/blob/master/Coach.py
"""

import ray
import logging
import os
import torch
import tqdm

from common import *
from game import game
from mcts import get_improved_policy

ray.init()

def start(
        args: dict,
        game: any,
        model_ref: ray.ObjectRef,
        replay_buffer,
):
    # Create self play workers and have them begin self-play
    self_play_runs = [
        SelfPlay.remote(args, game, model_ref).get_examples.remote(args.num_iterations_self_play, args.num_eps_self_play) for _ in range(args.num_workers_self_play)
    ]
    results = ray.get(self_play_runs)
    for result in results:
        replay_buffer.extend(result)

@ray.remote
class SelfPlay:
    def __init__(self, 
                 args: dict,
                 game: game.Game, 
                 model_ref: ray.ObjectRef
    ):
        self.game = game
        self.model = ray.get(model_ref)
        self.mcts = None
        self.args = args

    def get_examples(
            self,
            num_iterations_self_play: int,
            num_eps_self_play: int,
    ):
        # batch add training examples to replay buffer
        examples = []
        for i in range(num_iterations_self_play):
            for j in range(num_eps_self_play):
                # play games and add to replay buffer
                examples.extend(self.episode())

        return examples
    
    def episode(self):
        examples = []
        board = self.game.newBoard()
        step = 0

        while True:
            step += 1
            
            # Get the temperature (based on the move count)
            temp = int(step < self.args.temp_threshold)

            # Put first training example in examples (the reward can not yet be determined because the game is not finished.)
            improved_policy = get_improved_policy(board, self.args.num_iterations_mcts, self.args.num_workers_mcts, self.game, self.model, temp)

            # Append the improved policy to our training examples
            # TODO:fix
            examples.append([self.board_translator.encode(board), self.game.getCurrentPlayer(board), improved_policy.reshape(1, -1), None])

            # Now, sample an action from the improved policy
            # Switch to Tensor implementation
            a = torch.multinomial(improved_policy, num_samples = 1)

            # Get the next state with the action
            board = self.game.nextState(board, a)

            # Determine whether the game has ended  
            r = self.game.getRewards(board)

            # Put the reward in the training examples
            if r != None:
                logging.info("(self play) finished one episode of self play")
                # Return (board as tensor, policy, outcome)
                return [(x[0], x[2], -r * ((-1) ** (self.game.getCurrentPlayer(x[0]) != self.game.getCurrentPlayer(board)))) for x in examples]
    