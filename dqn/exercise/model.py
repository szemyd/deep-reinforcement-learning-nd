import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        "*** YOUR CODE HERE ***"
        # self.hidden_layer_1_size = int(round(state_size/2))
        # self.hidden_layer_2_size = int(round(self.hidden_layer_1_size/2))

        self.f1 = nn.Linear(state_size,  fc1_units)
        self.f2 = nn.Linear(fc1_units, fc2_units)
        self.f3 = nn.Linear(fc2_units, action_size)
    
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state.view(-1, state.shape[0])
        x = F.relu(self.f1(state))
        x = F.relu(self.f2(x))

        x = self.f3(x)

        return x

