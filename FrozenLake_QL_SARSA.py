#!/usr/bin/env python
# based on the code of basic_rl.py (v0.0.3)

from FrozenLake_QL_SARSA_options import *


# Start monitoring the simulation for OpenAI Gym
env = wrappers.Monitor(env, result_dir, force=True)

def learn(env,exploration_rate,beta, beta_inc):
	
	# Initialization of a Q-value table
	if args.initial_q_value=='optimistic':
		Q = q_mean + q_std * np.random.randn(n_s, n_a)
	else: # initialization to zeroes:
		Q = np.zeros([n_s, n_a])
	
	# Keeps track of useful statistics
	stats = EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))

	for i_episode in xrange(num_episodes):
		
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

	'''total_reward = 0

	for _ in range(100):
		env.render()
		s = env.reset()
		game_over = False

		while not game_over:
			action = np.argmax(Q[s, :])
			s, reward, game_over, _ = env.step(action)
			total_reward += reward'''
	
	return stats

if __name__ == "__main__":
	
	stats=learn(env,exploration_rate,beta,beta_inc)

	plot_episode_stats(stats, result_dir, smoothing_window=25)
