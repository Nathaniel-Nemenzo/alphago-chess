"""
Encapsulates the policy network implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Input tensor is 8x8x12, representing the board itself and 12 total pieces
"""
INPUT_CHANNELS = 12

class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace = True),
            nn.Conv2d(num_filters, num_filters, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_filters)
        )

    def forward(self, x):
        identity = x
        x = self.layers(x)
        x += identity
        x = F.relu(x, inplace = True)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self, num_filters = 128, num_res_layers = 11):
        super(PolicyNetwork, self).__init__()
        self.num_res_layers = num_res_layers
        self.num_filters = num_filters

        # conv -> bn -> residual -> policy head
        # 119 input channels (planes) in alphago board representations
        self.conv1 = nn.Conv2d(119, self.num_filters, kernel_size = 5, stride = 1, padding = 2)
        self.bn1 = nn.BatchNorm2d(self.num_filters)
        self.res_layers = self._build_res_layers()
        self.policy_head = nn.Sequential(
            nn.Conv2d(self.num_filters, 1, kernel_size = 1, stride = 1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace = True),
            nn.Flatten(),
            nn.Linear(64, 8 * 8 * 73),
            nn.LogSoftmax(dim = 1)
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
        x = self.policy_head(x)
        
        # return probability
        return x.view(-1, 8, 8, 73)

if __name__ == "__main__":
    model = PolicyNetwork(num_filters = 128)
    x = torch.zeros((3, 12, 8, 8))
    y = model(x)
    print(y.shape)