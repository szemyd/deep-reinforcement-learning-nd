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
LR = [0.01, 0.001, 0.0001]
actor_critic = [[60, 60], [96, 96], [120, 120], [160, 160], [240, 240]]
config_ddpg = [actor_critic, LR]

exp_config_ddpg = list(itertools.product(*config_ddpg))


