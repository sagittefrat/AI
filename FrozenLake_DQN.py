# based on https://github.com/andreimuntean/Simple-Neural-Q-Learning/blob/master/simple_neural_q_learning.py
from FrozenLake_DQN_options import *

env = wrappers.Monitor(env, result_dir, force=True)


def learn(env,exploration_rate):
  
    network=Network()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        # Keeps track of useful statistics
        stats = EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes))

        for i_episode in range(num_episodes):
            s = env.reset()
        

            for i_step in range(max_step):
                estimated_Qs = sess.run(network.y, {network.x: network.states[s:s + 1]})
                a = np.argmax(estimated_Qs, 1)[0]

                # Occasionally try a random action (explore).
                if np.random.rand(1) < exploration_rate:
                    a = env.action_space.sample()

                # Perform the action and observe its actual Q value.
                next_s, reward, done, _ = env.step(a)
                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = i_step

                observed_Q = reward + discount_rate * np.max(sess.run(network.y, {network.x: network.states[next_s:next_s + 1]}))
                
                # Measure error of initial estimation and learn from it.
                estimated_Qs[0, a] = observed_Q
                sess.run(network.train_step, {network.x: network.states[s:s + 1], network.y_: estimated_Qs})

                s = next_s

                if done:
                    break

            print('episode: {:d}  reward: {:g}  learning rate: {:g}'.format(i_episode + 1, reward, exploration_rate))

            exploration_rate *= exploration_rate_decay


        '''# Test the agent.
    
        total_reward = 0

        for _ in range(100):
            s = env.reset()
            game_over = False

            while not game_over:
                Qs = sess.run(y, feed_dict={x: states[s:s + 1]})
                action = np.argmax(Qs, 1)[0]
                s, reward, game_over, _ = env.step(action)
                total_reward += reward

      
        print('Average Reward:', total_reward / 100)'''

    return stats

if __name__ == "__main__":
    
    stats=learn(env,exploration_rate)

    plot_episode_stats(stats, result_dir, smoothing_window=25)