"""
Encapsulates worker to train the neural network based on examples collected from self-play.
"""

import torch
import queue
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from alphazero.replay_buffer import ReplayBuffer

class TrainingWorker:
    """Samples minibatches of moves from the replay buffer and perform SGD to fit the model. Every self.args.num_minibatches_to_send minibatches, send the model to the evaluator.
    """
    def __init__(self,
                 model: nn.Module,
                 replay_buffer: ReplayBuffer,
                 model_queue: queue.Queue,
                 args: dict):
        self.model = model
        self.replay_buffer = replay_buffer
        self.model_queue = model_queue
        self.args = args

    def start(self):
        # Set the model to training mode
        self.model.train()

        # Optimizer for model
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum = self.args.momentum)

        # Loss functions for policy and value
        value_loss_fn = nn.MSELoss()
        policy_loss_fn = nn.CrossEntropyLoss()

        # Train the model
        for i in range(self.args.num_training_iterations):
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
                predicted_policy, predicted_value = self.model(state)

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
                self.model_queue.put(self.model)

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
