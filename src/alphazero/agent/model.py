"""
Encapsulates the AlphaZero policy + value network implementation. AlphaZero uses one DCNN with two branches which computes
a policy (p) and value (v) and a Monte Carlo tree search to evaluate the state and update its action selection rule.

p is a learned probability distribution over all legal moves and v predicts the outcome of the game (-1, 0, 1) corresponding
to (lose, draw, win) based on the board representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# P1 pieces (6) + P2 pieces (6) + Repetitions (2)
_NUM_HISTORY_PLANES = 14

# Color (1) + Total Move Count (1) + P1 Castling (2) + P2 Castling (2) + No-progress count (1)
_NUM_ADDITIONAL_PLANES = 7

# Queen moves (56) + Knight Moves (8) + Underpromotions (9)
_NUM_POSSIBLE_MOVES = 73

_SIZE_CHESS_BOARD = 8

class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace = True),
            nn.Conv2d(num_filters, num_filters, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(num_filters)
        )

    def forward(self, x):
        identity = x
        x = self.layers(x)
        x += identity
        x = F.relu(x, inplace = True)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self, h = 8, num_filters = 256, num_res_layers = 11):
        super(PolicyNetwork, self).__init__()
        self.num_res_layers = num_res_layers
        self.num_filters = num_filters

        self.conv1 = nn.Conv2d(_NUM_HISTORY_PLANES * h + _NUM_ADDITIONAL_PLANES, self.num_filters, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(self.num_filters)
        self.res_layers = self._build_res_layers()

        # Policy head
        self.policy_head = nn.Sequential(
            # 1x1 conv; ReLU
            nn.Conv2d(self.num_filters, self.num_filters, kernel_size = 1, stride = 1, bias = False),
            nn.BatchNorm2d(self.num_filters),
            nn.ReLU(inplace = True),

            # 1x1 conv + biases; 
            nn.Conv2d(self.num_filters, _NUM_POSSIBLE_MOVES, kernel_size = 1, stride = 1, bias = True),

            # flatten to (8 * 8 * 73)
            nn.Flatten(),

            # apply softmax
            nn.LogSoftmax(dim = 1)
        )

        # Value head
        self.value_head = nn.Sequential(
            # 1x1 conv; ReLU
            nn.Conv2d(self.num_filters, 1, kernel_size = 1, stride = 1, bias = False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace = True),

            # flatten to (1, 64)
            nn.Flatten(),

            # linear layer; relu
            nn.Linear(_SIZE_CHESS_BOARD * _SIZE_CHESS_BOARD, 256),
            nn.ReLU(inplace = True),

            # linear layer to scalar
            nn.Linear(256, 1),

            # tanh
            nn.Tanh()
        )

    def _build_res_layers(self):
        res_layers = nn.ModuleList()
        for i in range(self.num_res_layers):
            res_layers.append(ResidualBlock(self.num_filters))
        return res_layers

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace = True)

        # pass through residual layers
        for res_layer in self.res_layers:
            x = res_layer(x)

        # pass through policy head
        x_policy = self.policy_head(x)
        
        # pass through value head
        x_value = self.value_head(x)

        # return probability distribution and approximated value
        return x_policy, x_value

if __name__ == "__main__":
    model = PolicyNetwork(num_filters = 256)
    x = torch.zeros((3, 119, 8, 8))
    y = model(x)
    print("policy shape: ", y[0].shape)
    print("value shape: ", y[1].shape)
    print(y[1])