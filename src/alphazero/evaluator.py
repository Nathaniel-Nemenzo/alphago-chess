import queue
import tqdm
import time
import os
import torch
import torch.nn as nn

from game.game import Game
from game.board_translator import BoardTranslator
from game.move_translator import MoveTranslator

class Evaluator:
    """Receives proposed new models and plays out self.args.num_evaluate_games between the new model and the old model and accepts any model that wins self.evaluation_threshold fraction of games or higher. Models that are accepted are published back to the self play worker.

    If there is no current model yet (when the evaluator is first ran), the evaluator will set the current model to the received model and publish the received model directly to the accepted model queue.
    """
    def __init__(self, 
                 initial_model: nn.Module,
                 board_translator: BoardTranslator,
                 move_translator: MoveTranslator,
                 new_model_queue: queue.Queue,
                 accepted_model_queue: queue.Queue,
                 game: Game,
                 args: dict):
        self.current_model = initial_model
        self.board_translator = board_translator
        self.move_translator = move_translator
        self.new_model_queue = new_model_queue
        self.accepted_model_queue = accepted_model_queue
        self.game = game
        self.args = args

    def start(self) -> None:
        """Start the evaluator. Checks for models in the new_model_queue and runs evaluate() to evaluate new models. If accepted, publishes the model to the accepted_model_queue.
        """
        while True:
            # Check for new models in the queue
            try:
                new_model_candidate = self.new_model_queue.get(timeout = self.args.new_model_queue_timeout)
            except queue.Empty:
                # Keep checking the queue if it's empty
                continue

            # Evaluate the new model against the current model
            if self.evaluate(self.current_model, new_model_candidate):
                # If the new model passes evaluation, update the current model to the new model
                self.current_model.load_state_dict(new_model_candidate.state_dict())

                # Publish the model to the accepted model queue
                self.accepted_model_queue.put(new_model_candidate)

                # Save the accepted model
                save_accepted_model(self.args.accepted_model_path, new_model_candidate)

    def evaluate(self, current_model, new_model) -> bool:
        """Plays the new model against the current model for self.args.num_evaluate_games. The old model will start self.args.num_evaluate_games / 2 games and the new model will start the same amount of games so both models experience both sides.G

        Returns:
            bool: Returns True if the percentage of wins by the new model is equal to or larger than self.evaluation_threshold, else False.
        """
        num_wins = 0
        first_player = current_model
        second_player = new_model
        halfway_point = self.args.num_evaluate_games // 2
        for i in tqdm(range(self.args.num_evaluate_games), desc = "evaluating models"):
            # Switch player order at halfway point
            if i == halfway_point:
                first_player, second_player = second_player, first_player

            # Play a game and get the result
            result = self.play_game(first_player, second_player)

            # Increment number of wins if new model wins as second player or if new model wins as first player
            if (i < halfway_point and result == -1) or (i >= halfway_point and result == 1):
                num_wins += 1
            

        return (num_wins / self.args.num_evaluate_games) >= self.args.evaluation_threshold

    def play_game(self, first_player: nn.Module, second_player: nn.Module) -> int:
        """Executes one episode of a game

        Returns:
            int: Returns 1 if first player has won, -1 if second player has won, and 0 if the game is tied.
        """
        # Start a new game
        board = self.game.getInitBoard()

        # Keep track of the current player
        current_player = first_player
        next_player = second_player

        # Keep going while the game has not ended
        while not self.game.getGameEnded(board):
            # Let the current player make a move
            tensor = self.board_translator.encode(board)

            # Forward the tensor representation through the model to get the policy
            policy, _ = current_player.forward(tensor)

            # Sort the policy in descending order and get the indices
            sorted_indices = policy.argsort(descending = True).flatten()

            # Try the actions one by one until the legal action with the highest probability is found
            action = None
            for index in sorted_indices:
                action = self.move_translator.decode(index)
                if self.game.checkIfValid(board, action):
                    break

            # Get the next state
            board = self.game.getNextState(board, action)

            # Switch players
            current_player, next_player = next_player, current_player

        # At this point, the game has ended
        result = board.result()
        if board.result() == "1-0": # White (first player) won
            return 1
        elif board.result() == "0-1": # Black (second player) won
            return -1
        else:
            return 0
        
def save_accepted_model(path, model):
    # Get the current timestamp
    now = int(time.time())

    # Create the folder and the file name
    file_name = f'{now}.pt'
    relative_dir = f'../../{path}'
    os.makedirs(relative_dir, exist_ok = True)

    # Save the model
    file_path = os.path.join(relative_dir, file_name)
    torch.save(model.state_dict(), file_path)