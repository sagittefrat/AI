#!/usr/bin/env python
# basic_rl.py (v0.0.3)
#
# New in v0.0.3
# - This version uses an average cumulative reward (ave_cumu_r)
#   instead of an average terminal reward (ave_terminal_r) to control the exploration rate.

from FrozenLake_QL_SARSA_options import *
import os
from gym import wrappers

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
exploration_rate_decay = args.exploration_rate_decay
q_mean = args.qmean
q_std = args.qstd

# Experimental setup
num_episodes = args.nepisode
max_step = args.maxstep
result_dir = 'results-QL-{0}-{1}-{2}'.format(env_type, policy_type,algorithm_type))
# Start monitoring the simulation for OpenAI Gym
#env = wrappers.Monitor(env, result_dir, force=True)

def learn(env,exploration_rate):
	
	# Initialization of a Q-value table
	if args.initial_q_value=='optimistic':
		Q = q_mean + q_std * np.random.randn(n_s, n_a)
	else: # initialization to zeroes:
		Q = np.zeros([n_s, n_a])
		 
	
	print "algorithm_type: {}".format(algorithm_type)
	print "policy_type: {}".format(policy_type)

	env.reset()		
	np.set_printoptions(precision=3, suppress=True)


	#create lists to contain total rewards and steps per episode
	rList = []
	# Keeps track of useful statistics
	stats = EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))

	for i_episode in xrange(num_episodes):
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

			if args.render_game:
				env.render()

			# Get a result of your action from the environment
			next_s, reward, done, info = env.step(a)

			# Update statistics
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = i_step

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
	
				if policy_type == 'epsilon_greedy':
					# exploration_rate is decayed expolentially
					exploration_rate = exploration_rate * exploration_rate_decay
				elif policy_type == 'softmax':
					# beta is increased linearly
					beta = beta + beta_inc

				print("\rStep {} @ Episode {}/{} ({})".format(i_step, i_episode + 1, num_episodes, stats.episode_rewards[i_episode] ))
				break

	# Test the agent.
	env = wrappers.Monitor(env, result_dir, force=True)
	total_reward = 0

	for _ in range(100):
		s = env.reset()
		game_over = False

		while not game_over:
			action = np.argmax(Q[s, :])
			s, reward, game_over, _ = env.step(action)
			total_reward += reward
	
	return stats

if __name__ == "__main__":
	
	stats=learn(env,exploration_rate)

	plot_episode_stats(stats, result_dir, smoothing_window=25)
