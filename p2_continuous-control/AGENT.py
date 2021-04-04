## Deep Deterministic Policy Gradients ##
from policy import Actor, Critic
import numpy as np

from constants import *

class DDPG_Agent():
    def __init__(self, state_size, action_size):
        super(DDPG_Agent, self).__init__()
        self.actor_local = Actor(state_size, action_size)
        self.actor_target = Actor(state_size, action_size)
        self.critic_local = Critic(state_size, action_size)
        self.critic_target = Critic(state_size, action_size)
    
    def act(self, state):
        state = torch.from_numpy(state).float().to(device)
        # action_probs = self.actor_local(state)
        # print(action_probs)
        # selected_index = np.random.choice(len(action_probs), size=1, p=action_probs.detach().numpy())
        self.actor_local.eval()
        with actor_local.no_grad():
            actions = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        return actions
    
    def step(self):

        pass

    def learn(self):    
        pass
    
    def soft_update(self, local, target, tau):
        """Soft update model parameters.
            θ_target = τ*θ_local + (1 - τ)*θ_target

            Params
            ======
                local_model: PyTorch model (weights will be copied from)
                target_model: PyTorch model (weights will be copied to)
                tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

