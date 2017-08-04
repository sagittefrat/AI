# based on https://github.com/dennybritz/reinforcement-learning/blob/35c5105fc403a883f869f429972b189360fea609/FA/Q-Learning%20with%20Value%20Function%20Approximation%20Solution.ipynb
from MountainCar_QL_options import *

env = wrappers.Monitor(env, result_dir, force=True)

def learn(env, Q_estimator, num_episodes, discount_factor, epsilon, epsilon_decay):
	"""
	Q-Learning algorithm for off-policy TD control using Function Approximation.
	Finds the optimal greedy policy while following an epsilon-greedy policy.
	
	Args:
		env: OpenAI environment.
		Q_estimator: Action-Value function estimator
		num_episodes: Number of episodes to run for.
		discount_factor: Lambda time discount factor.
		epsilon: Chance the sample a random action. Float betwen 0 and 1.
		epsilon_decay: Each episode, epsilon is decayed by this factor
	
	Returns:
		An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
	"""

	# Keeps track of useful statistics
	stats = EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes)) 

	policy=None
	for i_episode in range(num_episodes):
		
		
		policy = make_epsilon_greedy_policy(
			Q_estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)
		
		# Reset the environment and pick the first action
		s = env.reset()
		
		# Only used for SARSA, not Q-Learning
		next_a = None
		
		# One step in the environment
		for i_step in xrange(max_step):
						
			# Choose an action to take
			# If we're using SARSA we already decided in the previous step
			
			if next_a is None:
				a=select_action(Q_estimator,s,policy_type,beta,epsilon,i_episode,policy)
			else:
				a = next_a
			
			# Take a step
			next_s, reward, done, _ = env.step(a)
	
			# Update statistics
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = i_step
			
			# TD Update
			#if sarsa send next best action to get prediction:
			if algorithm_type=='sarsa': 
				next_a=select_action(Q_estimator,s,policy_type,beta,epsilon,i_episode,policy)
				q_values_next = Q_estimator.predict(next_s,next_a)
			else: q_values_next = Q_estimator.predict(next_s)
			
			# Use this code for Q-Learning
			# Q-Value TD Target
			td_target = reward + discount_factor * np.max(q_values_next)
			
			# Use this code for SARSA TD Target for on policy-training:
			# next_action_probs = policy(next_state)
			# next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)             
			# td_target = reward + discount_factor * q_values_next[next_action]
			
			# Update the function approximator using our target
			Q_estimator.update(s, a, td_target)
					
				
			if done:
				print("\rStep {} @ Episode {}/{} ({})".format(i_step, i_episode + 1, num_episodes, stats.episode_rewards[i_episode] ))
				break
				
			s = next_s
	
	return stats


if __name__ == "__main__":

Q_estimator = Q_Value()

# Note: For the Mountain Car we don't actually need an epsilon > 0.0
# because our initial estimate for all states is too "optimistic" which leads
# to the exploration of all states.
stats = learn(env, Q_estimator, num_episodes,discount_factor, epsilon,epsilon_decay)

plot_episode_stats(stats, result_dir, smoothing_window=25)


