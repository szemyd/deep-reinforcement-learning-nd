## Deep Deterministic Policy Gradients ##
from policy import Actor, Critic

class Agent():
    def __init__(self, state_size, action_size):
        super(Agent, self).__init__()
        self.actor_local = Actor(state_size, action_size)
        self.actor_target = Actor(state_size, action_size)
        self.critic_local = Critic(state_size, action_size)
        self.critic_target = Critic(state_size, action_size)
    
    def act(self, state):
        return self.actor_local(state)
    
    def step(self):
        pass

    def learn(self):
        pass
    