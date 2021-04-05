## Deep Deterministic Policy Gradients ##
from policy import Actor, Critic    # These are our models
import numpy as np
import random                       # Used for random seed
import copy                         # This is used for the mixing of target and local model parameters

from constants import *             # Capital lettered variables are constants from the constants.py file
from MEMORY import ReplayBuffer     # Our replaybuffer, where we store the experiences

import torch
import torch.nn.functional as F
import torch.optim as optim

class DDPG_Agent():
    def __init__(self, state_size, action_size, random_seed):
        super(DDPG_Agent, self).__init__()


        self.actor_local = Actor(state_size, action_size, random_seed).to(DEVICE)
        self.actor_target = Actor(state_size, action_size, random_seed).to(DEVICE)
        self.critic_local = Critic(state_size, action_size, random_seed).to(DEVICE)
        self.critic_target = Critic(state_size, action_size, random_seed).to(DEVICE)

        self.actor_opt = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        self.critic_opt = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        self.memory = ReplayBuffer(action_size, random_seed)

        self.seed = random.seed(random_seed)
    
    def act(self, state):
        state = torch.from_numpy(state).float().to(DEVICE)
        # action_probs = self.actor_local(state)
        # print(action_probs)
        # selected_index = np.random.choice(len(action_probs), size=1, p=action_probs.detach().numpy())
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        return actions
    
    def step(self, state, action, reward, next_state, done):
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences):

        states, actions, rewards, next_states, dones = experiences
 

        # ---                   Teach Critic (with TD)              --- #
        recommended_actions = self.actor_target(next_states)
        Q_nexts = self.critic_target(next_states, recommended_actions)
        Q_targets = (rewards + GAMMA * Q_nexts * (1 - dones))                 # This is what we actually got from experience
        Q_expected = self.critic_local(states, actions)                       # This is what we thought the expected return of that state-action is.
        critic_loss = CRITERION(Q_targets, Q_expected)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()


        # ---                   Teach Actor                          --- #
        next_actions = self.actor_local(states)
        # Here we get the value of each state-actions. 
        # This will be backpropagated to the weights that produced the action in the actor network. 
        # Large values will make weights stronger, smaller values (less expected return for that state-action) weaker
        actor_loss = -self.critic_local(states, next_actions).mean()            
        

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()


        # Mix model parameters in both Actor and Critic #
        self.soft_update(self.actor_local, self.actor_target) 
        self.soft_update(self.critic_local, self.critic_target) 
    
    def soft_update(self, local, target):
        """Soft update model parameters.
            θ_target = τ*θ_local + (1 - τ)*θ_target

            Params
            ======
                local_model: PyTorch model (weights will be copied from)
                target_model: PyTorch model (weights will be copied to)
                tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)

