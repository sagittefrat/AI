# based on https://github.com/bioothod/openai-mountain-car-v0/

#changed cr, max_step import state

import qlearn
from MountainCar_DQN_options import *

from copy import deepcopy


env_type=args.environment
env = gym.envs.make(env_type)
max_step=args.max_steps
#discount_factor=args.discount_factor
#epsilon=args.exploration_rate
#epsilon_decay=args.exploration_rate_decay
num_episodes=args.num_episode
algorithm_type='DQN'
policy_type=args.policy
result_dir = 'results/results-DN-{0}-{1}-{2}'.format(env_type, policy_type, algorithm_type)

class Mcar:
	def __init__(self, num_episodes, output_path,env,result_dir,max_step):
		self.num_episodes = num_episodes
		self.step = 0

		self.env = env
		self.env = gym.wrappers.Monitor(self.env, result_dir, force=True)

		num_feature  = self.env.observation_space.shape[0]

		self.obs_size = 2
		self.current_state = State(num_feature, self.obs_size)

		self.q = qlearn.Qlearn((num_feature*self.obs_size,), self.env.action_space.n, output_path)

	def new_state(self, obs):
		self.current_state.push_array(obs)
		self.current_state.complete()
		return deepcopy(self.current_state)

	def learn(self):
		last_rewards = []
		last_rewards_size = 100

		# Keeps track of useful statistics
		stats = EpisodeStats(
			episode_lengths=np.zeros(num_episodes),
			episode_rewards=np.zeros(num_episodes))

		for i_episode in range(self.num_episodes):
			observation = self.env.reset()
			s = self.new_state(observation)

			done = False

			for i_step in xrange(max_step):
				if args.render_game:
					self.env.render()

				a = self.q.get_action(s)
				new_observation, reward, done, info = self.env.step(a)
				# Update statistics
				stats.episode_rewards[i_episode] += reward
				stats.episode_lengths[i_episode] = i_step
				self.step += 1

				sn = self.new_state(new_observation)

				self.q.history.append((s, a, reward, sn, done), 1)
				self.q.learn()

				s = sn
				if done:
					break

			self.q.update_episode_stats(i_episode, stats.episode_rewards[i_episode])
			self.q.random_action_alpha_cap = self.q.ra_range_end - (self.q.ra_range_end - self.q.ra_range_begin) * (1. - float(i_step)/max_step)

			if len(last_rewards) >= last_rewards_size:
				last_rewards = last_rewards[1:]

			last_rewards.append(stats.episode_rewards[i_episode])
			mean = np.mean(last_rewards)


			print "%d episode, its reward: %d, total steps: %d, mean reward over last %d episodes: %.1f, std: %.1f" % (
					i_episode, stats.episode_rewards[i_episode], self.step, len(last_rewards), mean, np.std(last_rewards))

		self.env.close()
		return stats

if __name__ == "__main__":

	with tf.device('/cpu:0'):
		cp = Mcar(num_episodes, result_dir,env,result_dir, max_step)
		stats=cp.learn()
		plot_episode_stats(stats, result_dir, smoothing_window=25)
