import tensorflow as tf
import numpy as np

from network import *

BOARD_SIZE = 9

DEVICE = '/cpu:0' # '/gpu:0'

class NetworkPlayer:

    def __enter__(self):
        return self

    def __init__(self):
        self.session = tf.Session()

        create_network(self.session.graph, BOARD_SIZE)
        self.saver = tf.train.Saver()

        self.session.run(tf.initialize_all_variables())

        restore_checkpoint(self.saver, self.session)

        self.n_state = self.session.graph.get_collection('inputs')[0]
        self.n_policy, self.n_value = self.session.graph.get_collection('outputs')

    def get_max_action(self, state):
        action_policy = self.get_action_probs(state)
        return np.argmax(action_policy)

    def get_random_action(self, state):
        action_policy = self.get_action_probs(state)
        return np.random.choice(len(action_policy), p=action_policy)

    def get_action_probs(self, state):
        action_policy = self.get_action_probs_unpruned(state)
        ap = np.array(action_policy)
        ap *= np.array(state[2]).flatten()
        ap /= np.sum(ap)
        return ap

    def get_action_probs_unpruned(self, state):
        return self.session.run(self.n_policy, feed_dict={self.n_state: [state]})[0]

    def get_win_prediction(self, state):
        return self.session.run(self.n_value, feed_dict={self.n_state: [state]})[0]

    def close(self):
        self.session.close()

    def __exit__(self, type, value, tb):
        self.close()
