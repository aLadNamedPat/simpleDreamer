import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class Controller(nn.Module):
    def __init__(
        self,
        input_features,
        actions_dims,
        action_space=None,
    ):
        super(Controller, self).__init__()
        self.fc1 = nn.Linear(input_features, 30)
        self.LeakyReLU = nn.LeakyReLU()
        self.fc2 = nn.Linear(30, actions_dims)
        self.action_space = action_space

    def forward(
        self,
        x
    ):
        x = self.fc1(x)
        x = self.LeakyReLU(x)
        x = self.fc2(x)
        return x
    
    def random_action(
        self,
    ):
        raw = self.action_space.sample()

        if isinstance(raw, (int, np.integer)):
            return torch.tensor([[raw]], dtype=torch.long)
        arr = np.asarray(raw, dtype=np.float32)
        return arr
        # steer  = np.random.uniform(-1.0, 1.0)          # left / right
        # gas    = np.random.uniform(0.6, 1.0)           # always accelerate a bit
        # brake  = np.random.choice([0.0, np.random.uniform(0.2, 1.0)],
        #                         p=[0.8, 0.2])        # brake rarely
        # return np.array([steer, gas, brake], dtype=np.float32)