from collections import deque
import sys
import math
import numpy as np
# from util import writeToCsv

def interact(env, agent, num_episodes=20000, window=100, mode='train', render=1):
    """ Monitor agent's performance.
    
    Params
    ======
    - env: instance of OpenAI Gym's Taxi-v1 environment
    - agent: instance of class Agent (see Agent.py for details)
    - num_episodes: number of episodes of agent-environment interaction
    - window: number of episodes to consider when calculating average rewards

    Returns
    =======
    - avg_rewards: deque containing average rewards
    - best_avg_reward: largest value in the avg_rewards deque
    """
    print("New Simulation: num_episodes: ", num_episodes," | window: ", window, " | mode: ", mode)
    # initialize average rewards
    avg_rewards = deque(maxlen=num_episodes)
    # initialize best average reward
    best_avg_reward = -math.inf
    # initialize monitor for most recent rewards
    samp_rewards = deque(maxlen=window)
    # for each episode
    for i_episode in range(1, num_episodes+1):
        # begin the episode
        state = env.reset()
        # reset agent values
        agent.reset_episode(state, best_avg_reward)
        # initialize the sampled reward
        samp_reward = 0
        while True:
            # agent selects an action
            action = agent.select_action(state)
            # agent performs the selected action
            next_state, reward, done, _ = env.step(action)

            if reward>0:
                print(reward)

            
                
            # agent performs internal updates based on sampled experience
            agent.step(state, action, reward, next_state, done)
            # update the sampled reward
            samp_reward += reward
            # update the state (s <- s') to next time step
            state = next_state
           
            # if (i_episode == num_episodes):
            #     env.render()
            if mode == 'test' and (i_episode % math.floor(num_episodes/render)) == 0:
                env.render()
            if done:
                # save final sampled reward
                samp_rewards.append(samp_reward)
                if mode == 'test' and (i_episode % math.floor(num_episodes/render) == 0): 
                     env.close()  
                break
        
        if (i_episode >= window):
            
            # get average reward from last 100 episodes
            avg_reward = np.mean(samp_rewards)
            # append to deque
            avg_rewards.append(avg_reward)
            # update best average reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward

        
        # if (i_episode == num_episodes):
        #     writeToCsv("," + '%.3f' % best_avg_reward + "\n")
        # monitor progress
        print("\rEpisode {}/{} || Best average reward {}".format(i_episode, num_episodes, best_avg_reward), end="")
        sys.stdout.flush()
        # check if task is solved (according to OpenAI Gym)
        if best_avg_reward >= 9.7:
            print('\nEnvironment solved in {} episodes.'.format(i_episode), end="")
            break
        if i_episode == num_episodes: print('\n')
    
    env.close()
    return avg_rewards, best_avg_reward