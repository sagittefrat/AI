#!/usr/bin/env python

# python basic_rl_sarsa_QL.py -a sarsa -p random -lr 0.8 -ga 0.95 -n 2000 -initq zero : ~40% ~600ep
#62.45% episode ~300 ~8000ep
# python basic_rl_sarsa_QL.py -a q_learning -p random -lr 0.8 -ga 0.95 -n 2000 -initq zero : 50% ~300ep @same results as sarsa
# deep_q_learning ~100 
from __future__ import division
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
from collections import namedtuple
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Random seed
#np.random.RandomState(42)


import argparse
parser = argparse.ArgumentParser(description='Use SARSA/Q-learning algorithm with epsilon-greedy/softmax polciy.')
parser.add_argument('-a', '--algorithm', default='q_learning', choices=['sarsa', 'q_learning'],
                    help="Type of learning algorithm. (Default: q_learning)")
parser.add_argument('-p', '--policy', default='epsilon_greedy', choices=['epsilon_greedy', 'softmax', 'random'],
                    help="Type of policy. (Default: epsilon_greedy)")
parser.add_argument('-e', '--environment', default='FrozenLake-v0', choices=['Taxi-v2', 'Roulette-v0','FrozenLake-v0'],
                    help="Name of the environment provided in the OpenAI Gym. (Default: Taxi-v2)")
parser.add_argument('-n', '--nepisode', default='5000', type=int,
                    help="Number of episode. (Default: 20000)")
parser.add_argument('-lr', '--learning_rate', default='0.1', type=float,
                    help="Learning rate. (Default: 0.1)")
parser.add_argument('-be', '--beta', default='0.0', type=float,
                    help="Initial value of an inverse temperature. (Default: 0.0)")
parser.add_argument('-bi', '--betainc', default='0.01', type=float,
                    help="Linear increase rate of an inverse temperature. (Default: 0.01)")
parser.add_argument('-ga', '--discount_rate', default='0.99', type=float,
                    help="Discount rate. (Default: 0.99)")
parser.add_argument('-ep', '--exploration_rate', default='0.8', type=float,
                    help="Fraction of random exploration in the epsilon greedy. (Default: 0.8)")
parser.add_argument('-ed', '--exploration_rate_decay', default='0.995', type=float,
                    help="Decay rate of exploration_rate in the epsilon greedy. (Default: 0.995)")
parser.add_argument('-ms', '--maxstep', default='200', type=int,
                    help="Maximum step allowed in episode. (Default: 200)")
parser.add_argument('-qm', '--qmean', default='0.0', type=float,
                    help="Mean of the Gaussian used for initializing Q table. (Default: 0.0)")
parser.add_argument('-qs', '--qstd', default='1.0', type=float,
                    help="Standard deviation of the Gaussian used for initializing Q table. (Default: 1.0)")
parser.add_argument('-initq', '--initial_q_value', default='optimistic',choices=['optimistic','zero'],
                    help="initializing Q table values. (Default: optimistic)")
parser.add_argument('-resdir', '--results_dir', default= 'None')
parser.add_argument('-render', '--render_game', default=False,
                   help="Render the gym in window or not. (Default: False)")

args = parser.parse_args()

np.set_printoptions(precision=3, suppress=True)

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

def softmax(Q, beta=1.0):
    assert beta >= 0.0
    q_tilde = Q - np.max(Q)
    factors = np.exp(beta * q_tilde)
    return factors / np.sum(factors)

def select_a_with_softmax(s, Q, beta=1.0):
    prob_a = softmax(Q[s, :], beta=beta)
    cumsum_a = np.cumsum(prob_a)
    return np.where(np.random.rand() < cumsum_a)[0][0]

def select_a_with_epsilon_greedy(s, Q, epsilon=0.1):
    a = np.argmax(Q[s, :])
    if np.random.rand() < epsilon:
        a = np.random.randint(Q.shape[1])
    return a

def plot_value_function(V, title="Value Function"):
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))



def plot_episode_stats(stats, result_dir, smoothing_window=10):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    fig1.savefig(''+result_dir+'/episode.png')

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    fig2.savefig(''+result_dir+'/reward.png')
   
    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    fig3.savefig(''+result_dir+'/steps_episode.png')


def plot_score_board(history, window_size, policy_type, Q, n_s, n_a, result_dir ):


     fig, ax = plt.subplots(2, 2, figsize=[12, 8])
     # Number of steps
     ax[0, 0].plot(history[:, 0], history[:, 1], '.') 
     ax[0, 0].set_xlabel('Episode')
     ax[0, 0].set_ylabel('Number of steps')
     ax[0, 0].plot(history[window_size-1:, 0], running_average(history[:, 1], window_size))
     # Cumulative reward
     ax[0, 1].plot(history[:, 0], history[:, 2], '.') 
     ax[0, 1].set_xlabel('Episode')
     ax[0, 1].set_ylabel('Cumulative rewards')
     ax[0, 1].plot(history[:, 0], history[:, 4], '--')
     #ax[0, 1].plot(history[window_size-1:, 0], running_average(history[:, 2], window_size))
     # Terminal reward
     ax[1, 0].plot(history[:, 0], history[:, 3], '.')
     ax[1, 0].set_xlabel('Episode')
     ax[1, 0].set_ylabel('Terminal rewards')
     ax[1, 0].plot(history[window_size-1:, 0], running_average(history[:, 3], window_size))
     # Epsilon/Beta
     ax[1, 1].plot(history[:, 0], history[:, 5], '.') 
     ax[1, 1].set_xlabel('Episode')
     if policy_type == 'softmax':
          ax[1, 1].set_ylabel('Beta')
     elif policy_type == 'epsilon_greedy':
          ax[1, 1].set_ylabel('Epsilon')
     fig.savefig('./'+result_dir+'.png')

     print "Q value table:"
     print Q

     if policy_type == 'softmax':
          print "Action selection probability:"
          print np.array([softmax(q, beta=beta) for q in Q])
     elif policy_type == 'epsilon_greedy':
          print "Greedy action"
          greedy_action = np.zeros([n_s, n_a])
          greedy_action[np.arange(n_s), np.argmax(Q, axis=1)] = 1
          print greedy_action

def running_average(x, window_size, mode='valid'):
     return np.convolve(x, np.ones(window_size)/window_size, mode=mode)
