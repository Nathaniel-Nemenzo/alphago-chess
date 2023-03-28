import torch
import ray

from mcts import get_action

# TASK: implement parallel evaluation of the model

ray.init()

@ray.remote
class Evaluator:
    def __init(
            self,
            game: any,
            args: dict,
    ):
        self.game = game
        self.args = args

    def evaluate(
            self,
            current_model_ref: ray.ObjectRef,
            new_model_ref: ray.ObjectRef,
            num_iterations: int
    ):
        num_wins = 0
        first_player = ray.get(current_model_ref)
        second_player = ray.get(new_model_ref)
        for i in range(num_iterations):
            result = self.play_game(first_player, second_player)
            
            # check if new model is the second player
            if i % 2 == 0: 
                result *= -1

            if result == 1:
                num_wins += 1
            elif result == 0:
                num_wins += 0.5

            first_player, second_player = second_player, first_player

        return num_wins

    def play_game(
            self,
            first_player: torch.nn.Module,
            second_player: torch.nn.Module,
    ):
        board = self.game.get_init_board()
        current_player = first_player
        next_player = second_player

        while not self.game.get_game_ended(board):
            # get action
            action = get_action(
                board,
                self.args.mcts_simulations,
                self.args.num_workers,
                self.game,
                current_player,
                self.args.temp_threshold,
            )

            # update board
            board = self.game.get_next_state(board, action)

            # swap players
            current_player, next_player = next_player, current_player

        result = self.game.get_game_result(board)
        if result == "1-0":
            return 1
        elif result == "0-1":
            return -1
        else:
            return 0

def evaluate(
        args: dict,
        game: any,
        current_model_ref: ray.ObjectRef,
        new_model_ref: ray.ObjectRef,
):
    num_wins = 0

    # run evaluators
    evaluation_runs = [Evaluator.remote(
        game,
        args,
    ).evaluate.remote(current_model_ref, new_model_ref, args.num_games_per_worker_evaluator) for _ in range(args.num_workers_evaluator)]
    results = ray.get(evaluation_runs)
    num_wins = sum(results)

    # sum the number of wins
    return (num_wins / (args.num_games_per_worker_evaluator * args.num_workers_evaluator)) >= args.evaluation_threshold