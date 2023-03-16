import tqdm
import torch.nn as nn

from game.game import Game
from game.board_translator import BoardTranslator
from game.move_translator import MoveTranslator

class Evaluator:
    """Receives proposed new models and plays out self.args.num_evaluate_games between the new model and the old model and accepts any model that wins self.evaluation_threshold fraction of games or higher. Models that are accepted are published back to the self play worker.
    """
    def __init__(self, 
                 current_model: nn.Module, 
                 new_model: nn.Module,
                 board_translator: BoardTranslator,
                 move_translator: MoveTranslator,
                 game: Game,
                 args: dict):
        self.current_model = current_model
        self.new_model = new_model
        self.board_translator = board_translator
        self.move_translator = move_translator
        self.game = game
        self.args = args

    def evaluate(self) -> bool:
        """Plays the old model against the old model for self.args.num_evaluate_games. The old model will start self.args.num_evaluate_games / 2 games and the new model will start the same amount of games so both models experience both sides.G

        Returns:
            bool: Returns True if the percentage of wins by the new model is equal to or larger than self.evaluation_threshold, else False.
        """
        num_wins = 0
        first_player = self.current_model
        second_player = self.new_model
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
