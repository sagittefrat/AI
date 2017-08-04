import nn
from MountainCar_DQN_options import *


class action(object):
    def __init__(self, n):
        self.action = np.array([n])

    def value(self):
        return self.action

    def __hash__(self):
        return hash(self.action)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __str__(self):
        return str(self.action)

class Qlearn(object):
    def __init__(self, state_shape, action_space, output_path):
        self.alpha = 1
        self.gamma = 0.99
        self.random_action_alpha = 1
        self.random_action_alpha_cap = 1
        self.ra_range_begin = 0.05
        self.ra_range_end = 0.99
        self.total_actions = 0
        self.lam = 0.9
        self.history_size = 100000
        self.batch_size = 256

        self.action_space = action_space
        self.history = history(self.history_size)

        output_path += '/run.%d' % (time.time())
        self.summary_writer = tf.summary.FileWriter(output_path)

        self.main = nn.nn("main", state_shape[0], action_space, self.summary_writer)
        #self.follower = nn.nn("follower", state_shape[0], actions, self.summary_writer)

    def weighted_choice(self, ch):
        return np.random.choice()

    def get_action(self, s):
        self.total_actions += 1
        self.random_action_alpha = self.ra_range_begin + (self.random_action_alpha_cap - self.ra_range_begin) * math.exp(-0.0001 * self.total_actions)

        #self.random_action_alpha = 0.1
        random_choice = np.random.choice([True, False], p=[self.random_action_alpha, 1-self.random_action_alpha])

        if random_choice:
            return np.random.randint(0, self.action_space)

        q = self.main.predict(s.vector())
        return np.argmax(q[0])

    def learn(self):
        batch = self.history.sample(min(self.batch_size, self.history.size()))

        assert len(batch) != 0
        assert len(batch[0]) != 0
        assert len(batch[0][0].read()) != 0

        states_shape = (len(batch), len(batch[0][0].read()))
        states = np.ndarray(shape=states_shape)
        next_states = np.ndarray(shape=states_shape)

        q_shape = (len(batch), self.action_space)
        qvals = np.ndarray(shape=q_shape)
        next_qvals = np.ndarray(shape=q_shape)

        idx = 0
        for e in batch:
            s, a, r, sn, done = e

            states[idx] = s.read()
            next_states[idx] = sn.read()
            idx += 1

        qvals = self.main.predict(states)
        next_qvals = self.main.predict(next_states)

        for idx in range(len(batch)):
            e = batch[idx]
            s, a, r, sn, done = e

            qmax_next = np.amax(next_qvals[idx])
            if done:
                qmax_next = 0

            current_qa = qvals[idx][a]
            qsa = current_qa + self.alpha * (r + self.gamma * qmax_next - current_qa)
            qvals[idx][a] = qsa

        self.main.train(states, qvals)

        #if self.main.train_num % 10 == 0:
        #    self.follower.import_params(self.main.export_params())

    def update_episode_stats(self, episodes, reward):
        self.main.update_episode_stats(episodes, reward)
        #self.follower.update_episode_stats(episodes, reward)


class history_object(object):
    def __init__(self, o, w):
        self.o = o
        self.w = w

class history(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.history = deque()

        self.p = deque()
        self.p_sum = 0.0

    def clear(self):
        self.history = deque()
        self.p_sum = 0.0
        self.p = deque()

    def last(self, n):
        if n <= 0:
            return deque()

        start = 0
        if len(self.history) >= n:
            start = len(self.history) - n

        ret = deque()
        for i in range(start, len(self.history)):
            ret.append(self.history[i].o)
        
        return ret

    def size(self):
        return len(self.history)

    def full(self):
        return self.size() >= self.max_size

    def append(self, e, w):
        qlen = len(self.history) + 1
        if qlen > self.max_size:
            for i in range(qlen - self.max_size):
                self.p_sum -= self.history[0].w
                self.p.popleft()
                self.history.popleft()

        self.history.append(history_object(e, w))
        self.p.append(w)
        self.p_sum += w

    def sort(self):
        self.history = deque(sorted(self.history, key=lambda x: x.w))

    def get(self, idx):
        return self.history[idx].o

    def sample(self, size):
        idx = range(self.size())

        p = np.array(self.p) / self.p_sum
        ch = np.random.choice(idx, min(size, self.size()), p=p)

        ret = deque()
        for i in ch:
            ret.append(self.history[i].o)
        
        return ret

