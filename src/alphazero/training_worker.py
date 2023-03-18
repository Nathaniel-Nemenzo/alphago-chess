"""
Encapsulates worker to train the neural network based on examples collected from self-play.
"""

import logging
import torch
import queue
import torch.nn as nn

from common import *
from torch.utils.data import Dataset, DataLoader
from replay_buffer import ReplayBuffer

class TrainingWorker:
    """Samples minibatches of moves from the replay buffer and perform SGD to fit the model. Every self.args.num_minibatches_to_send minibatches, send the model to the evaluator.

    The training worker trains the latest accepted model. In addition, it always stays on (no training iterations). In the future, this will be limited to some training iteration limit over the entire training process.
    """
    def __init__(self,
                 device: torch.device,
                 shared_variables: dict,
                 replay_buffer: ReplayBuffer,
                 new_model_queue: queue.Queue,
                 args: dict):
        self.device = device
        self.shared_variables = shared_variables
        self.replay_buffer = replay_buffer
        self.new_model_queue = new_model_queue
        self.args = args

    def start(self):
        model = None
        optimizer = None

        # Loss functions for policy and value
        value_loss_fn = nn.MSELoss()
        policy_loss_fn = nn.CrossEntropyLoss()

        # Keep track of iterations
        i = 0

        # Train the model
        while True:
            # Check if there is a new model available.
            if self.shared_variables[TRAINING_SIGNAL]:
                # Update the model with the shared variable
                model = self.shared_variables[MODEL_TYPE]()

                # Put the model on the GPU
                model = model.to(self.device)
                logging.info('(training) created model and put model in gpu')

                model.load_state_dict(self.shared_variables[MODEL_STATE_DICT])
                logging.info('(training) loaded state dictionary')

                # Decrement the counter for the number of workers that have updated the model
                self.shared_variables[NUM_TRAINING_WORKERS] -= 1

                # Set the optimizer for model
                optimizer = torch.optim.SGD(model.parameters(), lr=self.args.learning_rate, momentum = self.args.momentum, weight_decay = 1e-4)

                # Set the model to training mode
                model.train()

                logging.info("(training): found new model and updated local model")

            # Sample minibatches from replay buffer
            sample = self.replay_buffer.sample(self.args.batch_size)

            # Create the DataLoader for sampled minibatches
            minibatch_dataset = MinibatchDataset(sample)
            data_loader = DataLoader(minibatch_dataset, batch_size = self.args.batch_size)

            # Perform training
            for state, improved_policy, value in data_loader:
                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                predicted_policy, predicted_value = model(state)

                # Calculate the loss
                policy_loss = policy_loss_fn(predicted_policy, improved_policy)
                value_loss = value_loss_fn(predicted_value, value)
                total_loss = policy_loss + value_loss

                # Backward pass
                total_loss.backward()

                # Optimize model
                optimizer.step()

            # Send the model to the evaluator every num_minibatches_to_send minibatches
            if i % self.args.num_minibatches_to_send == 0:
                logging.info(f"(training): sending model to evaluator on iteration {i}")
                self.new_model_queue.put(model)

            i += 1

class MinibatchDataset(Dataset):
    """Pytorch dataset that encapsulates minibatches that are sampled from the replay buffer used in training.

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, sample: list[tuple]):
        self.sample = sample

    def __len__(self):
        return len(self.sample)
    
    def __getitem__(self, idx):
        return self.sample[idx]
