""" 
AGENTS
======

DDPG 
actor  = [16, 16], [32, 32], [64, 64]
critic = [16, 16], [32, 32], [64, 64]

learning_rate = 0.0001, 0.001, 0.01
learning_rate = 0.001, 0.001, 0.01

3 x 3 x 3 = 27
"""


# --- ENVIRONMENT --- #

import numpy as np
import itertools

# --- AGENTS --- #
# DDPG
actor_critic = [[16], [32], [16, 16], [32, 32], [64, 64]]
config_ddpg = [actor_critic]

exp_config_ddpg = list(itertools.product(*config_ddpg))


