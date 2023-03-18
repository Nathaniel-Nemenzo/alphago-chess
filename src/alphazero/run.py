"""
Main entry point for running from command line
"""

import torch
import multiprocessing as mp
import logging

from game.chess.chess_model import AlphaZeroNetwork
from game.chess.chess_game import ChessGame
from game.chess.chess_board_translator import ChessBoardTranslator
from game.chess.chess_move_translator import ChessMoveTranslator

from common import *

from evaluator import Evaluator
from replay_buffer import ReplayBuffer
from self_play_worker import SelfPlayWorker
from training_worker import TrainingWorker

def configure_logger():
    logging.basicConfig(
        level=logging.DEBUG,  # Minimum log level to display
        format="%(asctime)s - %(levelname)s - %(threadName)s - %(message)s",  # Format of the log message
        handlers=[
            logging.StreamHandler(),  # Log to console
            logging.FileHandler("log.log")  # Log to file
        ]
    )

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
    
# Check if GPU is available
device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name()}")
else:
    device = torch.device("cpu")
    print("Using CPU")

if __name__ == "__main__":
    # Configure multiprocessing
    mp.set_start_method('spawn')

    # Configure logger
    configure_logger()

    # Set arguments
    args = dotdict({
        # Replay buffer
        'capacity': 1024,

        # Queue timeouts
        'new_model_queue_timeout': 300, # Check every 5 minutes
        'accepted_model_queue_timeout': 300, # 5 minutes

        # MCTS
        'cpuct': 1.5,
        'virtual_loss': 1.0,
        'num_mcts_simulations': 16,

        # Evaluator
        'num_evaluate_games': 16,
        'evaluation_threshold': 0.55,

        # Training Worker
        'learning_rate': 0.01,
        'momentum': 0.9,
        'batch_size': 5,
        'num_minibatches_to_send': 5,

        # Self-play worker
        'num_self_play_episodes': 10,

        # File paths for saving data / models
        'accepted_model_path': 'models',
        'training_example_path': 'data'
    })

    # Create manager for multiprocessing
    manager = mp.Manager()
    buffer = manager.Queue(args.capacity)
    shared_dict = manager.dict()

    # Create replay buffer
    replay_buffer = ReplayBuffer(buffer)

    # Create translators
    board_translator = ChessBoardTranslator(device)
    move_translator = ChessMoveTranslator(device)

    # Create game
    game = ChessGame(device, board_translator, move_translator)

    # Create model
    model = AlphaZeroNetwork()

    # Create shared dictionary
    manager = mp.Manager()
    shared_dict = manager.dict()

    # Set number of workers
    num_training_workers = 1
    num_self_play_workers = 1

    # Populate shared variables
    shared_dict[MODEL_TYPE] = type(model)
    shared_dict[MODEL_STATE_DICT] = model.state_dict()
    shared_dict[SELF_PLAY_SIGNAL] = False
    shared_dict[TRAINING_SIGNAL] = False
    shared_dict[NUM_SELF_PLAY_WORKERS] = num_self_play_workers
    shared_dict[NUM_TRAINING_WORKERS] = num_training_workers

    # Create new model queue
    new_model_queue = mp.Queue()

    # Create workers and evaluator
    self_play_worker = SelfPlayWorker(device, shared_dict, game, replay_buffer, board_translator, move_translator, args)
    training_worker = TrainingWorker(device, shared_dict, replay_buffer, new_model_queue, args)
    evaluator = Evaluator(device, shared_dict, model, board_translator, move_translator, new_model_queue, game, args)

    # Kick off threads for self-play, training, and evaluation
    self_play_process = mp.Process(target=self_play_worker.start)
    training_process = mp.Process(target=training_worker.start)
    evaluator_process = mp.Process(target=evaluator.start)

    # Start processes
    self_play_process.start()
    training_process.start()
    evaluator_process.start()

    # Wait for processes to finish (they never will)
    self_play_process.join()
    training_process.join()
    evaluator_process.join()