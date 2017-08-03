
#!/usr/bin/env python

import gym
import numpy as np
import os
import random
from gym import wrappers
import time
from MountainCar_DQN_options import *

GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.5  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 10000  # experience replay buffer size

policy_type=args.policy
algorithm_type=args.algorithm
num_episodes=args.nepisode
env_type=args.environment
max_step=args.max_step
optimizer=args.optimizer
learning_rate = args.learning_rate
result_dir=args.checkpoint_dir
mode=args.mode
batch_size=args.batch_size
env = gym.make(env_type)
result_dir = 'results-DQN-{0}-{1}-{2}'.format(env_type, policy_type,algorithm_type)
env = wrappers.Monitor(env, result_dir, force=True)

# Random seed
np.random.RandomState(2)
def replay(replay_buffer,state_input,sess,Q_value):

	minibatch = random.sample(replay_buffer, batch_size)
	state_batch = [data[0] for data in minibatch]
	action_batch = [data[1] for data in minibatch]
	reward_batch = [data[2] for data in minibatch]
	next_state_batch = [data[3] for data in minibatch]

	y_batch = []

	Q_value_batch = sess.run(Q_value,feed_dict={state_input: next_state_batch})

	for i in range(batch_size):
		done = minibatch[i][4]
		if done:
			y_batch.append(reward_batch[i])
		else:
			y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[
					i]))

	return (y_batch, action_batch, state_batch)

def dqn():

	replay_buffer = deque()
	epsilon = INITIAL_EPSILON
	state_dim = env.observation_space.shape[0]

	action_dim = env.action_space.n
	tf.reset_default_graph()
	state_input = tf.placeholder("float", [None, state_dim])
	
	Q_value = inference(state_input,state_dim,action_dim)

	action_input = tf.placeholder("float", [None, action_dim])
	y_input = tf.placeholder("float", [None])
	Q_action = tf.reduce_sum(tf.multiply(Q_value, action_input), reduction_indices=1)
	loss = tf.reduce_mean(tf.square(y_input - Q_action))
	
	print("Use the optimizer: {}".format(optimizer))
	if optimizer == "sgd":
		optim = tf.train.GradientDescentOptimizer(learning_rate)
	elif optimizer == "adadelta":
		optim = tf.train.AdadeltaOptimizer(learning_rate)
	elif optimizer == "adagrad":
		optim = tf.train.AdagradOptimizer(learning_rate)
	elif optimizer == "adam":
		optim = tf.train.AdamOptimizer(learning_rate)
	elif optimizer == "ftrl":
		optim = tf.train.FtrlOptimizer(learning_rate)
	elif optimizer == "rmsprop":
		optim = tf.train.RMSPropOptimizer(learning_rate)
	else:
		print("Unknow optimizer: {}, exit now".format(optimizer))
		exit(1)

	global_step = tf.Variable(0, name='global_step', trainable=False)
	train_op = optim.minimize(loss, global_step=global_step)

	if not os.path.exists(result_dir):
		os.makedirs(result_dir)
	checkpoint_file = result_dir + "/checkpoint.ckpt"
	init_op = tf.global_variables_initializer()
	saver = tf.train.Saver()
	tf.summary.scalar('loss', loss)

	with tf.Session() as sess:
		summary_op = tf.summary.merge_all()
		writer = tf.summary.FileWriter(args.tensorboard_dir, sess.graph)
		sess.run(init_op)

		if mode == "train":
			# Restore from checkpoint if it exists
			ckpt = tf.train.get_checkpoint_state(result_dir)
			if ckpt and ckpt.model_checkpoint_path:
				print("Restore model from the file {}".format(ckpt.model_checkpoint_path))
				saver.restore(sess, ckpt.model_checkpoint_path)
			# Keeps track of useful statistics
			stats = EpisodeStats(
			episode_lengths=np.zeros(num_episodes),
			episode_rewards=np.zeros(num_episodes))

			for i_episode in range(num_episodes):
				# Start new epoisode to train
				print("Start to train with episode: {}".format(i_episode))
				state = env.reset()
				loss_value = -999

				for i_step in xrange(max_step):

					# Get action from exploration and exploitation
					if random.random() <= epsilon:
						action = random.randint(0, action_dim - 1)
					else:
						Q_value_value = sess.run(Q_value, feed_dict={state_input: [state]})[0]
						action = np.argmax(Q_value_value)

					next_state, reward, done, _ = env.step(action)

					# Update statistics
					stats.episode_rewards[i_episode] += reward
					stats.episode_lengths[i_episode] = i_step

					# Get new state add to replay experience queue
					one_hot_action = np.zeros(action_dim)
					one_hot_action[action] = 1
					replay_buffer.append((state, one_hot_action, reward, next_state, done))
					
					if len(replay_buffer) > REPLAY_SIZE:
						replay_buffer.popleft()

					# Get batch replay experience to train
					if len(replay_buffer) > batch_size:
						(y_batch, action_batch, state_batch)=replay(replay_buffer,state_input,sess,Q_value)

						_, loss_value, i_step = sess.run(
								[train_op, loss, global_step],
								feed_dict={
										y_input: y_batch,
										action_input: action_batch,
										state_input: state_batch
								})
			
					state = next_state
					if done:
						break

				'''# Validate for some episode		
				if i_episode % args.episode_to_validate == 0:
					print("Global step: {}, the loss: {}".format(i_step, loss_value))

					state = env.reset()
					total_reward = 0

					for j_step in xrange(max_step):
						if args.render_game:
							env.render()
						Q_value2 = sess.run(Q_value, feed_dict={state_input: [state]})
						action = np.argmax(Q_value2[0])
						state, reward, done, _ = env.step(action)
						total_reward += reward
						if done:
							break

					print("Eposide: {}, total reward: {}".format(i_episode, total_reward))'''

			# End of training process
			saver.save(sess, checkpoint_file, global_step=i_step)

		elif mode == "untrained":
			total_reward = 0
			state = env.reset()

			for i in xrange(max_step):
				if args.render_game:
					time.sleep(0.1)
					env.render()
				action = env.action_space.sample()
				next_state, reward, done, _ = env.step(action)
				total_reward += reward

				if done:
					print("End of untrained because of done, reword: {}".format(total_reward))
					break

		elif mode == "inference":
			print("Start to inference")

			# Restore from checkpoint if it exists
			ckpt = tf.train.get_checkpoint_state(result_dir)
			if ckpt and ckpt.model_checkpoint_path:
				print("Restore model from the file {}".format(
						ckpt.model_checkpoint_path))
				saver.restore(sess, ckpt.model_checkpoint_path)
			else:
				print("Model not found, exit now")
				exit(0)

			total_reward = 0
			state = env.reset()

			for i in xrange(max_step):
				time.sleep(0.1)
				if args.render_game:
					env.render()
				Q_value_value = sess.run(Q_value, feed_dict={state_input: [state]})[0]
				action = np.argmax(Q_value_value)
				next_state, reward, done, _ = env.step(action)
				state = next_state
				total_reward += reward

				if done:
					print("End of inference because of done, reword: {}".format(total_reward))
					break

		else:
			print("Unknown mode: {}".format(mode))

	print("End of playing game")
	return stats

if __name__ == "__main__":

	stats=dqn()
	plot_episode_stats(stats, result_dir, smoothing_window=25)
