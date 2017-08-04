#!/usr/bin/env python

import gym
from gym import wrappers
import tensorflow as tf
import json, sys, os
from os import path
import random
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque,namedtuple


import argparse
parser = argparse.ArgumentParser(description='Use SARSA/Q-learning algorithm with epsilon-greedy/softmax polciy.')
parser.add_argument('-a', '--algorithm', default='q_learning', choices=['sarsa', 'q_learning'],
                    help="Type of learning algorithm. (Default: q_learning)")
parser.add_argument('-p', '--policy', default='softmax', choices=['epsilon_greedy', 'softmax', 'random'],
                    help="Type of policy. (Default: epsilon_greedy)")
parser.add_argument('-e', '--environment', default='MountainCar-v0', choices=['CartPole-v0','MountainCar-v0'],
                    help="Name of the environment provided in the OpenAI Gym. (Default: CartPole-v0)")
parser.add_argument('-n', '--num_episode', default='1000', type=int,
                    help="Number of episode. (Default: 1000)")
parser.add_argument('-lr', '--learning_rate', default='0.5', type=float,
                    help="Learning rate. (Default: 0.1)")
parser.add_argument('-be', '--beta', default='0.3', type=float,
                    help="Initial value of an inverse temperature. (Default: 0.0)")
parser.add_argument('-bi', '--betainc', default='0.01', type=float,
                    help="Linear increase rate of an inverse temperature. (Default: 0.01)")
parser.add_argument('-ga', '--discount_rate', default='0.99', type=float,
                    help="Discount rate. (Default: 0.99)")
parser.add_argument('-ep', '--exploration_rate', default='0.8', type=float,
                    help="Fraction of random exploration in the epsilon greedy. (Default: 0.8)")
parser.add_argument('-ed', '--exploration_rate_decay', default='0.995', type=float,
                    help="Decay rate of exploration_rate in the epsilon greedy. (Default: 0.995)")
parser.add_argument('-ms', '--max_steps', default='200', type=int,
                    help="Maximum step allowed in episode. (Default: 200)")
parser.add_argument('-ka', '--kappa', default='0.01', type=float,
                    help="Weight of the most recent cumulative reward for computing its running average. (Default: 0.01)")
parser.add_argument('-qm', '--qmean', default='0.0', type=float,
                    help="Mean of the Gaussian used for initializing Q table. (Default: 0.0)")
parser.add_argument('-qs', '--qstd', default='1.0', type=float,
                    help="Standard deviation of the Gaussian used for initializing Q table. (Default: 1.0)")
parser.add_argument('-initq', '--initial_q_value', default='zero',choices=['optimistic','zero'],
                    help="initializing Q table values. (Default: optimistic)")
parser.add_argument('-resdir', '--results_dir', default= 'None')
parser.add_argument('-render', '--render_game', default=False,
                   help="Render the gym in window or not. (Default: False)")
parser.add_argument('-dumb', '--dumb_reward', default= 'False')

args = parser.parse_args()
EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])


np.set_printoptions(precision=3, suppress=True)

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


#def running_average(x, window_size, mode='valid'):
    #return np.convolve(x, np.ones(window_size)/window_size, mode=mode)

class State(object):
    def __init__(self, shape, size):
        self.steps = deque()
        self.shape = shape
        self.size = size
        self.value = None

        for i in range(size):
            self.push_zeroes()

    def push_zeroes(self):
        self.push_array(np.zeros(self.shape))

    def push_array(self, step_array):
        assert self.shape == step_array.shape[0]

        if len(self.steps) == self.size:
            self.steps.popleft()

        self.steps.append(step_array)

    def complete(self):
        self.value = np.concatenate(self.steps)

    def read(self):
        return self.value

    def reshape(self, rows, cols):
        return self.value.reshape(rows, cols)

    def vector(self):
        return self.value.reshape(1, self.value.shape[0])


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





