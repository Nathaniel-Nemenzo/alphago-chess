"""
Encapsulates worker to generate training examples from self-play using neural network-aided Monte Carlo tree search.
"""

import ray
import torch

from common import *
from game import game
from mcts import get_improved_policy

ray.init()

def start(
        args: dict,
        game: any,
        model_ref: ray.ObjectRef,
        replay_buffer: ray.ActorRef,
):
    # Create self play workers and have them begin self-play
    self_play_runs = [
        SelfPlay.remote(args, game, model_ref).get_examples.remote(args.num_iterations_self_play, args.num_eps_self_play) for _ in range(args.num_workers_self_play)
    ]
    results = ray.get(self_play_runs)
    replay_buffer = ray.get_actor(replay_buffer)
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
    
            temp = int(step < self.args.temp_threshold)

            # get improved policy
            improved_policy = get_improved_policy(board, self.args.num_iterations_mcts, self.args.num_workers_mcts, self.game, self.model, temp)

            # add training example to examples
            # here, we append the policy and the player who is playing
            examples.append([self.game.get_tensor_representation_of_board(board), self.game.get_current_player(board), improved_policy.reshape(1, -1), None])

            a = torch.multinomial(improved_policy, num_samples = 1)
            board = self.game.get_next_state(board, a)

            # returns 1 if white won and -1 if black won
            r = self.game.get_rewards(board)

            # check for a draw
            if r == 0:
                return [(x[0], x[2], 0) for x in examples]
            else:                
                # if we win on this state, then the player that made the PRIOR move won. (it's not possible to make a move and lose in chess)
                loser = self.game.get_current_player(board)
                return [(x[0], x[2], -1 if x[1] == loser else 1) for x in examples]
    