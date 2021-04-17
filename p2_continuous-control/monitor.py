import numpy as np
import matplotlib.pyplot as plt
import csv
import time
from helper import get_constant_string

def calculate_moving_avarage(scores, num_agent=1, scores_window=SCORES_WINDOW):
    single_agent_returns = scores
    moving_avarages=[np.convolve(scores[i], np.ones(scores_window)/scores_window, mode='valid') for i in range(num_agent)]
    
    return moving_avarages


def render_save_graph(scores, scores_window = 0, num_agent = 1):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(np.arange(1, len(scores)+1), scores)

    if scores_window > 0:
        moving_avarages = calculate_moving_avarage([scores], num_agent, scores_window)

        for i_agent in range(len(moving_avarages)):
            plt.plot(np.arange(len(moving_avarages[i_agent]) + scores_window)[scores_window:], moving_avarages[i_agent], 'g-')

    plt.ylabel('Score')
    plt.xlabel('Episode #')

    hyperparameter_string, for_filename  = get_constant_string()

    plt.title(hyperparameter_string)
    plt.savefig("Experiments/Figure_{}_{}.jpg".format(time.strftime("%Y-%m-%d_%H%M"), for_filename), bbox_inches='tight')
    
    plt.show()


def save_scores(scores):
    hyperparameter_string, for_filename  = get_constant_string()

    with open("Experiments/Scores_{}_{}.csv".format(time.strftime("%Y-%m-%d_%H%M"), for_filename), 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(scores)

    print("Scores saved!")