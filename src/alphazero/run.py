"""
Main entry point for running from command line
"""

from evaluator import Evaluator
from replay_buffer import ReplayBuffer
from self_play_worker import SelfPlayWorker
from training_worker import TrainingWorker

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
    
if __name__ == "__main__":
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
        'num_training_iterations': 3,
        'batch_size': 5,
        'num_minibatches_to_send': 5,

        # Self-play worker
        'num_self_play_iterations': 1,
        'num_self_play_episodes': 10,

        # File paths for saving data / models
        'accepted_model_path': 'models',
        'training_example_path': 'data'
    })

    # Create replay buffer