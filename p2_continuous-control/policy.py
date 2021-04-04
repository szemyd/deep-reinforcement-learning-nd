import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_space, action_space, hidden_layer_param = [50, 50]):
        super(Actor, self).__init__()

        self.fc_in = nn.Linear(state_space, hidden_layer_param[0])
        self.hidden_layers = [nn.Linear(hidden_layer_param[i], hidden_layer_param[i+1]) for i in range(len(hidden_layer_param)-1)]
        self.fc_out = nn.Linear(hidden_layer_param[-1], action_space)

        self.activation = nn.ReLU()


    def forward(self, state):
        x = state.view(-1)
        x = self.activation(self.fc_in(x))

        for hidden_layer in self.hidden_layers:
            x = activation(hidden_layer(x))

        x = self.activation(self.fc_out) 

        return F.tanh(x)

class Critic(nn.Module):
    def __init__(self, state_space, action_space, hidden_layer_param = [50, 50]):
        super(Critic, self).__init__()

        self.fc_in = nn.Linear(state_space, hidden_layer_param[0])
        self.hidden_layers = [nn.Linear(hidden_layer_param[i], hidden_layer_param[i+1]) for i in range(len(hidden_layer_param)-1)]
        self.fc_out = nn.Linear(hidden_layer_param[-1], action_space)

        self.activation = nn.ReLU()

    def forward(self, state):
        x = state.view(-1)
        x = self.activation(self.fc_in(x))

        for hidden_layer in self.hidden_layers:
            x = activation(hidden_layer(x))

        x = self.activation(self.fc_out) 

        return F.tanh(x)