#!/usr/bin/env python
# based on the code of basic_rl.py (v0.0.3)

from FrozenLake_QL_SARSA_options import *


# Start monitoring the simulation for OpenAI Gym
env = wrappers.Monitor(env, result_dir, force=True)

def learn(env,exploration_rate,beta, beta_inc):
	
	# Initialization of a Q-value table
	Q=initilize_q()
	
	# Keeps track of useful statistics
	stats = EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))

	for i_episode in xrange(num_episodes):
		
		# Start a new episode and sample the initial state
		s = env.reset()

		# Select the first action in this episode
		a=select_a(policy_type, s, Q, i_episode, n_a, beta=beta, epsilon=exploration_rate)
		

		for i_step in xrange(max_step):
			#print 'Q\n',Q
			
			
			if args.render_game:
				env.render()

			# Get a result of your action from the environment
			next_s, reward, done, info = env.step(a)

			# Update statistics
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = i_step

			# Select an action
			next_a=select_a(policy_type, next_s, Q, i_episode, n_a, beta=beta, epsilon=exploration_rate)            
			#print 's,a,next_s,next_a', s,a,next_s,next_a
			
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
				# you can speed up the processif giving negative reward when done and not achieving goal but this is solution to this specific problem
				#print 'done'
				
				(beta,exploration_rate)=update_policy_parameters(policy_type,exploration_rate=exploration_rate,exploration_rate_decay=exploration_rate_decay,beta=beta,beta_inc=beta_inc)
				#print exploration_rate
				#raw_input()
				print("\rStep {} @ Episode {}/{} ({})".format(i_step, i_episode + 1, num_episodes, stats.episode_rewards[i_episode] ))
				break

	
	return stats

if __name__ == "__main__":
	
	stats=learn(env,exploration_rate,beta,beta_inc)

	plot_episode_stats(stats, result_dir, smoothing_window=25)
