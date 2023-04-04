"""
Encapsulates worker to train the neural network based on examples collected from self-play.
"""

import ray
import torch
import torch.nn as nn

from ray import train


from common import *
from torch.utils.data import Dataset, DataLoader
from replay_buffer import ReplayBuffer

ray.init()

def train(
        config: dict
):
    trainer = train.Trainer(backend="torch", use_gpu=True, num_workers=config["num_workers_train"])

    trainer.start()
    results = trainer.run(train_epoch, config=config)
    trainer.shutdown()
    return results

def train_epoch(
        config: dict
):
    # epoch info
    epoch_info = {"num_steps": 0, "sum_loss": 0}
    
    # dataset
    replay_buffer = ray.get(config["replay_buffer_ref"])
    dataset = replay_buffer.get_dataset()
    data_loader = DataLoader(dataset, batch_size=config["batch_size"])

    # training params
    model = ray.get(config["model_ref"])
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum = 0.9, weight_decay = 1e-4)
    policy_loss_fn = torch.nn.CrossEntropyLoss()
    value_loss_fn = torch.nn.MSELoss()

    # wrap model and dataset to distribute
    model = train.torch.prepare_model(model)
    data_loader = train.torch.prepare_data_loader(data_loader)

    # training loop
    for epoch in range(config["batches_per_epoch"]):
        
        # train batches
        for batch_idx, state, improved_policy, value in enumerate(data_loader):
            batch = (state, improved_policy, value)
            batch_info = train_batch(
                batch,
                batch_idx,
                optimizer,
                model,
                policy_loss_fn,
                value_loss_fn,
            )

            epoch_info["num_steps"] += 1
            epoch_info["sum_loss"] += batch_info["loss"]

    return {"loss": epoch_info["sum_loss"] / epoch_info["num_steps"]}

def train_batch(
        batch,
        batch_idx,

        # training params
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        policy_loss_fn: torch.nn.CrossEntropyLoss,
        value_loss_fn: torch.nn.MSELoss,

):
    state, improved_policy, value = batch

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

    return {f"loss for batch {batch_idx}": total_loss.item()}
