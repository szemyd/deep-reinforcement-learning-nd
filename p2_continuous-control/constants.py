
import torch
import torch.nn as nn
import torch.nn.functional as F

# General Training parameters #
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     # Training on GPU or CPU

# Replay Buffer parameters #
BUFFER_SIZE = int(1e5)          # Replay Buffer size
UPDATE_EVERY = 4                # Define how often the target model gets exchanged by the local model (episode num). Not used in DDPG, because we gradually mix the two models
BATCH_SIZE = 128                # Minibatch size

# Learning parameters #
GAMMA = 0.99                    # Discount rate
LR_ACTOR = 1e-4                 # Learning rate for Actor optimization
LR_CRITIC = 1e-3                # Learning rate for Critic optimization
CRITERION = F.mse_loss        # What criterion to use when comparing expected return to target return

# Target Mixin probability
TAU = 1e-3 