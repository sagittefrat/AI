#https://github.com/dennybritz/reinforcement-learning/blob/35c5105fc403a883f869f429972b189360fea609/FA/Q-Learning%20with%20Value%20Function%20Approximation%20Solution.ipynb

import gym
import sys
from gym import wrappers
from MountainCar_QL_options import *


matplotlib.style.use('ggplot')
env_type=args.environment
# initializing the problem:
env = gym.envs.make(env_type)
algorithm_type = args.algorithm
policy_type = args.policy
max_step=args.max_step
discount_factor=args.discount_factor
epsilon=args.exploration_rate
epsilon_decay=args.exploration_rate_decay
num_episodes=args.nepisode
beta=args.beta

result_dir = 'results/results-QL-{0}-{1}-{2}'.format(env_type, policy_type,algorithm_type)
env = wrappers.Monitor(env, result_dir, force=True)

(featurizer, scaler)=preprocess(env)

class Q_Value():
	"""
	Value Function approximator. This is basiclly the handler of the Q table
	"""
	
	def __init__(self):
		# We create a separate model for each action in the environment's
		# action space. Alternatively we could somehow encode the action
		# into the features, but this way it's easier to code up.
		self.actions = []
		for _ in xrange(env.action_space.n):
			act = SGDRegressor(learning_rate="constant")
			# We need to call partial_fit once to initialize the model
			# or we get a NotFittedError when trying to make a prediction
			# This is quite hacky.
			act.partial_fit([self.featurize_state(env.reset())], [0])
			self.actions.append(act)
	
	def featurize_state(self, state):
		"""
		Returns the featurized representation for a state.
		"""
		scaled = scaler.transform([state])
		featurized = featurizer.transform(scaled)
		return featurized[0]
	
	def predict(self, s, a=None):
		"""
		Makes value function predictions.
		
		Args:
			s: state to make a prediction for
			a: (Optional) action to make a prediction for
			
		Returns
			If an action a is given this returns a single number as the prediction.
			If no action is given this returns a vector or predictions for all actions
			in the environment where pred[i] is the prediction for action i.
			
		"""
		features = self.featurize_state(s)
		if not a:
			return np.array([act.predict([features])[0] for act in self.actions])
		else:
			return self.actions[a].predict([features])[0]
	
	def update(self, s, a, td_target):
		"""
		Updates the q_estimator parameters for a given state and action towards
		the target y.
		"""
		features = self.featurize_state(s)
		self.actions[a].partial_fit([features], [td_target])

def select_action(Q,s,policy_type,beta,exploration_rate,i_episode,policy):
	action_probs = policy(s)
	if (1 - exploration_rate) <= np.random.uniform(0, 1):
		return np.random.choice(np.arange(len(action_probs)))
	else:
		# Select the first action in this episode

		if policy_type == 'softmax':
			a = select_a_with_softmax(Q_estimator.predict(s), beta=beta)
		elif policy_type == 'epsilon_greedy':
			a = np.random.choice(np.arange(len(action_probs)), p=action_probs)
		elif policy_type == 'random':
			a = np.argmax(Q_estimator.predict(s) + np.random.randn(1,len(action_probs))*(1./(i_episode+1)))
		else:
			raise ValueError("Invalid policy_type: {}".format(policy_type))
		return a


def make_epsilon_greedy_policy(Q_estimator, epsilon, action_space_n):
	"""
	Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
	
	Args:
		Q_estimator: An estimator that returns q values for a given state
		epsilon: The probability to select a random action . float between 0 and 1.
		nA: Number of actions in the environment.
	
	Returns:
		A function that takes the observation as an argument and returns
		the probabilities for each action in the form of a numpy array of length action_space_n.
	
	"""
	def policy_fn(observation):
		A = np.ones(action_space_n, dtype=float) * epsilon / action_space_n
		q_values = Q_estimator.predict(observation)
		best_action = np.argmax(q_values)
		A[best_action] += (1.0 - epsilon)
		return A

	return policy_fn



def q_learning(env, Q_estimator, num_episodes, discount_factor, epsilon, epsilon_decay):
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


Q_estimator = Q_Value()


# Note: For the Mountain Car we don't actually need an epsilon > 0.0
# because our initial estimate for all states is too "optimistic" which leads
# to the exploration of all states.
stats = q_learning(env, Q_estimator, num_episodes,discount_factor, epsilon,epsilon_decay)


#plotting.plot_cost_to_go_mountain_car(env, Q_stimator)
plot_episode_stats(stats, result_dir, smoothing_window=25)


