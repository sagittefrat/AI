
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
import gym
from gym import wrappers
import sys


# Random seed
np.random.RandomState(42)

# this defaults belongs to MountainCar_QL.py - No neural Network
import argparse
parser = argparse.ArgumentParser(description='Use SARSA/Q-learning algorithm with epsilon-greedy/softmax policy.')
parser.add_argument('-a', '--algorithm', default='q_learning', choices=['sarsa', 'q_learning'],
					help="Type of learning algorithm. (Default: q_learning)")
parser.add_argument('-p', '--policy', default='epsilon_greedy', choices=['epsilon_greedy', 'softmax', 'random'],
					help="Type of policy. (Default: epsilon_greedy)")
parser.add_argument('-e', '--environment', default='MountainCar-v0', choices=['MountainCar-v0', 'CartPole-v0','Acrobot-v0'],
					help="Name of the environment provided in the OpenAI Gym. (Default: 'MountainCar-v0')")
parser.add_argument('-n', '--nepisode', default='1000', type=int,
					help="Number of episode. (Default: 100)")
parser.add_argument('-lr', '--learning_rate', default='0.1', type=float,
					help="Learning rate. (Default: 0.1)")
parser.add_argument('-be', '--beta', default='0.9', type=float,
					help="Initial value of an inverse temperature. (Default: 0.0)")
parser.add_argument('-bi', '--betainc', default='0.01', type=float,
					help="Linear increase rate of an inverse temperature. (Default: 0.01)")
parser.add_argument('-ga', '--discount_factor', default='1.0', type=float,
					help="Discount factor. (Default: 1.0)")
parser.add_argument('-ep', '--exploration_rate', default='0.1', type=float,
					help="Fraction of random exploration in the epsilon greedy. (Default: 0.0)")
parser.add_argument('-ed', '--exploration_rate_decay', default='1.0', type=float,
					help="Decay rate of exploration_rate in the epsilon greedy. (Default: 1.0)")
parser.add_argument('-ms', '--max_step', default='1000', type=int,
				   help="Maximum step allowed in episode. (Default: 1000)")
args = parser.parse_args() 


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
learning_rate = args.learning_rate

result_dir = 'results/results-QL-{0}-{1}-{2}-nepisode{3}-lr{4}'.format(env_type, policy_type,algorithm_type,num_episodes,learning_rate)
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


def preprocess(env):
	# Feature Preprocessing: Normalize to zero mean and unit variance
	# We use a few samples from the observation space to do this
	observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
	scaler = sklearn.preprocessing.StandardScaler()
	scaler.fit(observation_examples)

	# Used to converte a state to a featurizes represenation - remember countinues state space
	# We use RBF kernels with different variances to cover different parts of the space
	featurizer = sklearn.pipeline.FeatureUnion([
			("rbf1", RBFSampler(gamma=5.0, n_components=100)),
			("rbf2", RBFSampler(gamma=2.0, n_components=100)),
			("rbf3", RBFSampler(gamma=1.0, n_components=100)),
			("rbf4", RBFSampler(gamma=0.5, n_components=100))
			])
	featurizer.fit(scaler.transform(observation_examples))
	return (featurizer, scaler)


def softmax(Q, beta=1.0):
	assert beta >= 0.0

	q_tilde = Q - np.max(Q)
	#print q_tilde
	factors = np.exp(beta * q_tilde)
	#print factors, factors / np.sum(factors)
	return factors / np.sum(factors)

def select_a_with_softmax(q_values, beta=1.0):
	#print q_values
	prob_a = softmax(q_values, beta=beta)
	#print prob_a
	#raw_input()
	cumsum_a = np.cumsum(prob_a)
	return np.where(np.random.rand() < cumsum_a)[0][0]

def select_a_with_epsilon_greedy(s, Q, epsilon=0.1):
	a = np.argmax(Q[s, :])
	if np.random.rand() < epsilon:
		a = np.random.randint(Q.shape[1])
	return a

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

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


