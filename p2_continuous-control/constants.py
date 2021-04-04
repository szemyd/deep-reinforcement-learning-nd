
import torch

BUFFER_SIZE = int(1e5)  # replay buffer size
UPDATE_EVERY = 4
BATCH_SIZE = 128        # minibatch size
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

