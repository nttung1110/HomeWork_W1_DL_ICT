import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class SingleLayerMLP(nn.Module):
    def __init__(self, n_units, n_input, is_categorical):
        super(SingleLayerMLP, self).__init__()
        self.linear_1 = nn.Linear(n_input, n_units)
        if is_categorical:
            self.linear_2 = nn.Linear(n_units, 2)
        else:
            self.linear_2 = nn.Linear(n_units, 1)

    def forward(self, inputs):
        output = F.relu(self.linear_1(inputs))
        output = self.linear_2(output).squeeze()

        return output

class ThreeLayerMLP(nn.Module):
    def __init__(self, n_input, drop_out_rate=None):
        super(ThreeLayerMLP, self).__init__()
        self.linear_1 = nn.Linear(n_input, 32)
        self.linear_2 = nn.Linear(32, 64)
        self.linear_3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 2)

    def forward(self, inputs):
        output = F.relu(self.linear_1(inputs))
        output = F.relu(self.linear_2(output))
        output = F.relu(self.linear_3(output))
        output = self.out(output).squeeze()

        return output