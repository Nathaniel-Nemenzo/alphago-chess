"""
Encapsulates worker that performs self-play and writes game data to a file to be trained on at a later time
"""

import sys
sys.path.append('../helpers')
sys.path.append('../agent')

import os
import torch
from helpers.dataset import ChessDataset
from agent.policy import PolicyNetwork
from torch.utils.data import DataLoader

def start():
    # Set CUDA device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    model = PolicyNetwork()
    model.to(device)
    dataset = ChessDataset('/home/nathaniel/Code/alphago-chess/data/lichess_elite_database') # todo: remove
    return SLWorker.start(model, dataset, device)

class SLWorker:
    def __init__(self, model, dataset, device):
        self.model = model
        self.dataset = dataset
        self.device = device

    def start(self, batch_size = 32, num_epochs = 10, learning_rate = 0.003, save_interval = 500, save_dir = '../models/'):
        """
        Start actual training the SL model
        """
        optimizer = torch.optim.SGD(self.model.parameters(), lr = learning_rate, nesterov = True)
        criterion = torch.nn.CrossEntropyLoss()
        dataloader = DataLoader(self.dataset, batch_size = batch_size, shuffle = True)
        total_steps = 0
        for epoch in range(num_epochs):
            for i, (X_batch, y_batch) in enumerate(dataloader):
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                total_steps += 1
                print(total_steps)

                if total_steps % save_interval == 0:
                    model_file = os.path.join(save_dir, 'model_{}.pt'.format(total_steps))
                    torch.save(self.model.state_dict(), model_file)
                if i % 100 == 0:
                    print("Epoch {}/{}, Batch {}/{}, Loss: {}".format(epoch+1, num_epochs, i+1, len(dataloader), loss.item()))


    def get_games(self):
        """
        Get all games from SL game directory
        """
        return NotImplemented

def get_game():
    """
    Get all games from a PGN game file
    """
    return NotImplemented