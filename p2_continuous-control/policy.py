import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_space, action_space, random_seed = 32, hidden_layer_param = [400, 300] ):
        super(Actor, self).__init__()

        self.fc_in = nn.Linear(state_space, hidden_layer_param[0])
        self.hidden_layers = [nn.Linear(hidden_layer_param[i], hidden_layer_param[i+1]) for i in range(len(hidden_layer_param)-1)]
        self.fc_out = nn.Linear(hidden_layer_param[-1], action_space)

        self.activation = F.relu

        self.seed = torch.manual_seed(random_seed)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.fc_in.weight.data.uniform_(*hidden_init(self.fc_in))
        for hidden_layer in self.hidden_layers:
            hidden_layer.weight.data.uniform_(*hidden_init(hidden_layer))
        self.fc_out.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = self.activation(self.fc_in(state))

        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))

        x = self.activation(self.fc_out(x)) 

        return F.tanh(x)

class Critic(nn.Module):
    def __init__(self, state_space, action_space, random_seed = 32, hidden_layer_param = [400, 300] ):
        super(Critic, self).__init__()

        self.fc_in = nn.Linear(state_space, hidden_layer_param[0])

        self.hidden_layers = [nn.Linear(hidden_layer_param[i], hidden_layer_param[i+1]) for i in range(len(hidden_layer_param)-1)]
        self.hidden_layers[0] = nn.Linear(hidden_layer_param[0] + action_space, hidden_layer_param[1])

        
        # Critic throws back a single value (output = 1), which is the estimated value of the given state # 
        self.fc_out = nn.Linear(hidden_layer_param[-1], 1)

        self.activation = F.relu

        self.seed = torch.manual_seed(random_seed)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc_in.weight.data.uniform_(*hidden_init(self.fc_in))
        for hidden_layer in self.hidden_layers:
            hidden_layer.weight.data.uniform_(*hidden_init(hidden_layer))
        self.fc_out.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = self.activation(self.fc_in(state))
        x = torch.cat((xs, action), dim=1)

        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))

        return self.fc_out(x)