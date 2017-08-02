#!/usr/bin/env python
# basic_rl.py (v0.0.3)
#
# New in v0.0.3
# - This version uses an average cumulative reward (ave_cumu_r)
#   instead of an average terminal reward (ave_terminal_r) to control the exploration rate.

from options import *

import gym
#from gym import wrappers
import numpy as np
import os
import matplotlib.pyplot as plt

def main():

	env_type = args.environment
	algorithm_type = args.algorithm
	policy_type = args.policy

	# Random seed
	np.random.RandomState(42)

	# Selection of the problem
	env = gym.envs.make(env_type)
	
	# Constraints imposed by the environment
	n_a = env.action_space.n
	n_s = env.observation_space.n

	# Meta parameters for the RL agent
	learning_rate = args.learning_rate
	beta = args.beta
	beta_inc = args.betainc
	discount_rate = args.discount_rate
	exploration_rate = args.exploration_rate
	exploration_rate_decay = args.exploration_ratedecay
	q_mean = args.qmean
	q_std = args.qstd

	# Experimental setup
	num_episode = args.nepisode
	#print "n_episode ", num_episode
	max_step = args.maxstep

	# Running average of the cumulative reward, which is used for controlling an exploration rate
	# (This idea of controlling exploration rate by the terminal reward is suggested by JKCooper2)
	# See https://gym.openai.com/evaluations/eval_xSOlwrBsQDqUW7y6lJOevQ
	kappa = args.kappa
	ave_cumu_r = None
	
	# Initialization of a Q-value table
	if args.initial_q_value=='optimistic':
		Q = q_mean + q_std * np.random.randn(n_s, n_a)
	else: # initialization to zeroes:
		Q = np.zeros([n_s, n_a])
		 
	# Initialization of a list for storing simulation history
	history = []
	
	print "algorithm_type: {}".format(algorithm_type)
	print "policy_type: {}".format(policy_type)

	env.reset()		
	np.set_printoptions(precision=3, suppress=True)

	'''if args.results_dir=='None':
		result_dir='results-{0}-{1}-{2}'.format(env_type, algorithm_type, policy_type)
	else: result_dir =args.results_dir'''
	result_dir = 'results-{0}-{1}-{2}'.format(env_type, algorithm_type, policy_type)
	
	# Start monitoring the simulation for OpenAI Gym
	#env.monitor.start(result_dir, force=True)
	env=gym.wrappers.Monitor(env,result_dir,force=True)

	#create lists to contain total rewards and steps per episode
	rList = []
	for i_episode in xrange(num_episode):
		r_tot = 0
		
		# Reset a cumulative reward for this episode
		cumu_r = 0

		# Start a new episode and sample the initial state
		s = env.reset()

		# Select the first action in this episode
		if policy_type == 'softmax':
			a = select_a_with_softmax(s, Q, beta=beta)
		elif policy_type == 'epsilon_greedy':
			a = select_a_with_epsilon_greedy(s, Q, epsilon=exploration_rate)
		elif policy_type == 'random':
			a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i_episode+1)))
		else:
			raise ValueError("Invalid policy_type: {}".format(policy_type))

		for i_step in xrange(max_step):

			# Get a result of your action from the environment
			next_s, reward, done, info = env.step(a)

			# Modification of reward
			# CAUTION: Changing this part of the code in order to get a fast convergence
			# is not a good idea because it is essentially changing the problem setting itself.
			# This part of code was kept not to get fast convergence but to show the 
			# influence of a reward function on the convergence speed for pedagogical reason.
			#if done & (r == 0):
			#    # Punishment for falling into a hall
			#    r = 0.0
			#elif not done:
			#    # Cost per step
			#    r = 0.0

			# Update a cummulative reward
			cumu_r = reward + discount_rate * cumu_r

			# Select an action
			if policy_type == 'softmax':
				next_a = select_a_with_softmax(next_s, Q, beta=beta)
			elif policy_type == 'epsilon_greedy':
				next_a = select_a_with_epsilon_greedy(next_s, Q, epsilon=exploration_rate)
			elif policy_type == 'random':
				next_a = np.argmax(Q[next_s,:] + np.random.randn(1,env.action_space.n)*(1./(i_episode+1)))
			else:
				raise ValueError("Invalid policy_type: {}".format(policy_type))            

			# Calculation of TD error
			if algorithm_type == 'sarsa':
				delta = reward + discount_rate * Q[next_s, next_a] - Q[s, a]
			elif algorithm_type == 'q_learning':
				delta = reward + discount_rate * np.max(Q[next_s, :]) - Q[s, a]
			else:
				raise ValueError("Invalid algorithm_type: {}".format(algorithm_type))

			# Update a Q value table
			Q[s, a] += learning_rate * delta

			r_tot += reward
			s = next_s
			a = next_a

			if done:

				# Running average of the terminal reward, which is used for controlling an exploration rate
				# (This idea of controlling exploration rate by the terminal reward is suggested by JKCooper2)
				# See https://gym.openai.com/evaluations/eval_xSOlwrBsQDqUW7y6lJOevQ
				kappa = 0.01
				if ave_cumu_r == None:
					ave_cumu_r = cumu_r
				else:
					ave_cumu_r = kappa * cumu_r + (1 - kappa) * ave_cumu_r
				
				if cumu_r > ave_cumu_r:
					# Bias the current policy toward exploitation
					
					if policy_type == 'epsilon_greedy':
						# exploration_rate is decayed expolentially
						exploration_rate = exploration_rate * exploration_rate_decay
					elif policy_type == 'softmax':
						# beta is increased linearly
						beta = beta + beta_inc
						
				if policy_type == 'softmax':
					print "Episode: {0}\t Steps: {1:>4}\tCumuR: {2:>5.2f}\tTermR: {3}\tAveCumuR: {4:.3f}\tBeta: {5:.3f}".format(
						i_episode, i_step, cumu_r, reward, ave_cumu_r, beta)
					history.append([i_episode, i_step, cumu_r, reward, ave_cumu_r, beta])
				elif policy_type == 'epsilon_greedy':
					print "Episode: {0}\t Steps: {1:>4}\tCumuR: {2:>5.2f}\tTermR: {3}\tAveCumuR: {4:.3f}\tEpsilon: {5:.3f}".format(
						i_episode, i_step, cumu_r, reward, ave_cumu_r, exploration_rate)
					history.append([i_episode, i_step, cumu_r, reward, ave_cumu_r, exploration_rate])
				elif policy_type == 'random':
					print "Episode: {0}\t Steps: {1:>4}\tCumuR: {2:>5.2f}\tTermR: {3}\tAveCumuR: {4:.3f}\tr_tot: {5:.3f}".format(
						i_episode, i_step, cumu_r, reward, ave_cumu_r, r_tot)
				else:
					raise ValueError("Invalid policy_type: {}".format(policy_type))

				break
		rList.append(r_tot)

	# Stop monitoring the simulation for OpenAI Gym
	#env.monitor.close()

	history = np.array(history)
	window_size = 100

	print "Percent of succesful episodes: " + str(sum(rList)/num_episode) + "%"
	plt.plot(rList)
	plt.show()
	plot_score_board(history, window_size, policy_type, Q, n_s, n_a, result_dir )


if __name__ == "__main__":
	main()