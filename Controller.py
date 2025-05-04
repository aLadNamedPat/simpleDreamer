import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class Controller(nn.Module):
    def __init__(
        self,
        input_features,
        actions_dims,
    ):
        super(Controller, self).__init__()
        self.fc1 = nn.Linear(input_features, 30)
        self.LeakyReLU = nn.LeakyReLU()
        self.fc2 = nn.Linear(30, actions_dims)

    def forward(
        self,
        x
    ):
        x = self.fc1(x)
        x = self.LeakyReLU(x)
        x = self.fc2(x)
        return x
    
    