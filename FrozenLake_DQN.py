import gym
import numpy as np
import tensorflow as tf
from gym import wrappers
from collections import namedtuple
from matplotlib import pyplot as plt
import pandas as pd
from FrozenLake_DQN_options import *


env_type = args.environment
algorithm_type = args.algorithm
policy_type = args.policy

# Random seed
np.random.RandomState(42)

# Selection of the problem
env = gym.envs.make(env_type)
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
result_dir = 'results/results-DQN-{0}-{1}-{2}'.format(env_type, policy_type,algorithm_type)
env = wrappers.Monitor(env, result_dir, force=True)

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

def learn(env,exploration_rate):
    # Agent can be in one of 16 states.
    states = np.identity(16)

    x = tf.placeholder(shape=[None, env.observation_space.n], dtype=tf.float32)
    W = tf.Variable(tf.random_uniform([env.observation_space.n, env.action_space.n], 0, 0.1))

    # Estimated Q values for each action.
    y = tf.matmul(x, W)

    # Observed Q values (well only one action is observed, the rest remain equal to 'y').
    y_ = tf.placeholder(shape=[env.observation_space.n, env.action_space.n], dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(y_ - y))
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        # Keeps track of useful statistics
        stats = EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes))

        for i_episode in range(num_episodes):
            s = env.reset()
        

            for i_step in range(max_step):
                estimated_Qs = sess.run(y, {x: states[s:s + 1]})
                a = np.argmax(estimated_Qs, 1)[0]

                # Occasionally try a random action (explore).
                if np.random.rand(1) < exploration_rate:
                    a = env.action_space.sample()

                # Perform the action and observe its actual Q value.
                next_s, reward, done, _ = env.step(a)
                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = i_step

                observed_Q = reward + discount_rate * np.max(sess.run(y, {x: states[next_s:next_s + 1]}))
                
                # Measure error of initial estimation and learn from it.
                estimated_Qs[0, a] = observed_Q
                sess.run(train_step, {x: states[s:s + 1], y_: estimated_Qs})

                s = next_s

                if done:
                    break

            print('episode: {:d}  reward: {:g}  leatning rate: {:g}'.format(i_episode + 1, reward, exploration_rate))

            exploration_rate *= exploration_rate_decay


        # Test the agent.
    
        total_reward = 0

        for _ in range(100):
            s = env.reset()
            game_over = False

            while not game_over:
                Qs = sess.run(y, feed_dict={x: states[s:s + 1]})
                action = np.argmax(Qs, 1)[0]
                s, reward, game_over, _ = env.step(action)
                total_reward += reward

      
        print('Average Reward:', total_reward / 100)

    return stats

if __name__ == "__main__":
    
    stats=learn(env,exploration_rate)

    plot_episode_stats(stats, result_dir, smoothing_window=25)