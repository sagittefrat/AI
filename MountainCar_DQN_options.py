
from collections import deque,namedtuple
import tensorflow as tf
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

import argparse
parser = argparse.ArgumentParser(description='Use SARSA/Q-learning algorithm with epsilon-greedy/softmax polciy.')
parser.add_argument('-a', '--algorithm', default='q_learning', choices=['sarsa', 'q_learning'],
                    help="Type of learning algorithm. (Default: q_learning)")
parser.add_argument('-e', '--environment', default='MountainCar-v0', choices=['MountainCar-v0', 'CartPole-v0','Acrobot-v0'],
					help="Name of the environment provided in the OpenAI Gym. (Default: 'MountainCar-v0')")
parser.add_argument('-p', '--policy', default='epsilon_greedy', choices=['epsilon_greedy', 'softmax', 'random'],
                    help="Type of policy. (Default: epsilon_greedy)")
parser.add_argument('-lr', '--learning_rate', default='0.01', type=float,
					help="Learning rate. (Default: 0.001)")
parser.add_argument('-ga', '--discount_rate', default='0.9', type=float,
                    help="Discount rate. (Default: 0.9)")
parser.add_argument('-n', '--nepisode', default='1000', type=int,
					help="Number of episode. (Default: 200)")
parser.add_argument('-ms', '--max_step', default='500', type=int,
				   help="Maximum step allowed in episode. (Default: 500)")
parser.add_argument('-bs', '--batch_size', default='32', type=int,
				   help="indicates batch size in a single cpu. (Default: 32)")
parser.add_argument('-cpk', '--checkpoint_dir', default='./checkpoint/',
					help="indicates the checkpoint dirctory. (Default: ./checkpoint/)")
parser.add_argument('-tensor', '--tensorboard_dir', default='./tensorboard/',
					help="indicates training output. (Default: ./tensorboard/)")
parser.add_argument('-optim', '--optimizer', default='adam',
					help="indicates optimizer metodology. (Default: adam)")
parser.add_argument('-val', '--episode_to_validate', default='1', type=int,
				   help="Steps to validate and print loss. (Default: 1)")
parser.add_argument('-bn', '--batch_normalization', default='False',
				   help="Use batch normalization or not. (Default: False)")
parser.add_argument('-mod', '--mode', default='train',choices=['inference','train','untrained'],
					help="Opetion mode. (Default: train)")
parser.add_argument('-render', '--render_game', default=False,
				   help="Render the gym in window or not. (Default: False)")
arser.add_argument('-ep', '--exploration_rate', default='0.5', type=float,
                    help="Fraction of random exploration in the epsilon greedy. (Default: 0.5)")
parser.add_argument('-ed', '--exploration_rate_decay', default='0.995', type=float,
                    help="Decay rate of exploration_rate in the epsilon greedy. (Default: 0.995)")
args = parser.parse_args() 


# Define the model
def inference(inputs,state_dim,action_dim):
	# The inputs is [BATCH_SIZE, state_dim], outputs is [BATCH_SIZE, action_dim]
	hidden1_unit_number = 20
	with tf.variable_scope("fc1"):
		weights = tf.get_variable("weight",
									[state_dim, hidden1_unit_number],
									initializer=tf.random_normal_initializer())
		bias = tf.get_variable("bias",
								[hidden1_unit_number],
								initializer=tf.random_normal_initializer())
		layer = tf.add(tf.matmul(inputs, weights), bias)

	# Batch normalization
	if args.batch_normalization:
		mean, var = tf.nn.moments(layer, axes=[0])
		scale = tf.get_variable("scale",
								hidden1_unit_number,
								initializer=tf.random_normal_initializer())
		shift = tf.get_variable("shift",
								hidden1_unit_number,
								initializer=tf.random_normal_initializer())
		epsilon = 0.001
		layer = tf.nn.batch_normalization(layer, mean, var, shift, scale,
																			epsilon)

	layer = tf.nn.relu(layer)

	with tf.variable_scope("fc2"):
		weights = tf.get_variable("weight",
								[hidden1_unit_number, action_dim],
								initializer=tf.random_normal_initializer())
		bias = tf.get_variable("bias",
								[action_dim],
								initializer=tf.random_normal_initializer())
	layer = tf.add(tf.matmul(layer, weights), bias)

	return layer
	

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




