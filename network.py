import tensorflow as tf
import math
from datetime import datetime

WEIGHT_STDDEV = 0.01
BIAS_CONSTANT = 0.01
LEARNING_RATE = 1e-3

def weight_variable(shape, num_inputs):
    return tf.Variable(tf.truncated_normal(shape, stddev=WEIGHT_STDDEV), name='weights')

def bias_variable(shape):
    return tf.Variable(tf.constant(BIAS_CONSTANT, shape=shape), name='bias')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def relu_convolutional_layer(x, W, layer_num):
    with tf.name_scope('HiddenConv-L{}'.format(layer_num)):
        W_conv = weight_variable(W, W[-2])
        b_conv = bias_variable([W[-1]])
    return tf.nn.relu(conv2d(x, W_conv) + b_conv)

class Shared_VP_Network(object):

    def __init__(self, board_size, tfsession):
        with tf.device('/cpu:0'):
            self.state_input = tf.placeholder(tf.float32, shape=[None, 3, board_size, board_size], name='State_Input')

            state_board = tf.transpose(self.state_input, perm=[0, 2, 3, 1]) # transpose from [?, 3, 9, 9] array to [?, 9, 9, 3] array

            h_conv1 = relu_convolutional_layer(state_board, [5, 5, 3, 32], 1)
            h_conv2 = relu_convolutional_layer(h_conv1, [4, 4, 32, 64], 2)
            h_conv3 = relu_convolutional_layer(h_conv2, [3, 3, 64, 64], 3)

            h_conv3_flat = tf.reshape(h_conv3, [-1, board_size * board_size * 64])

            with tf.name_scope('HiddenFC-L4'):
                W_fcl4 = weight_variable([board_size * board_size * 64, 512], board_size * board_size * 64)
                b_fcl4 = bias_variable([512])
            h_fcl4 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fcl4) + b_fcl4)

            # Policy Network Output

            with tf.name_scope('PolicyOutputFC-L5'):
                W_pfcl5 = weight_variable([512, board_size * board_size], 512)
                b_pfcl5 = bias_variable([board_size * board_size])

            self.policy_network = tf.nn.softmax(tf.matmul(h_fcl4, W_pfcl5) + b_pfcl5)

            # Value Network Output

            with tf.name_scope('ValueOutputFC-L5'):
                W_vfcl5 = weight_variable([512, 1], 512)
                b_vfcl5 = bias_variable([1])

            self.value_network = tf.matmul(h_fcl4, W_vfcl5) + b_vfcl5

            # Training

            # Algorithm S3 from https://arxiv.org/abs/1602.01783
            # Google Deepmind -- Asynchronous Methods for Deep Reinforcement Learning
            # Asynchronous Advantage Actor Critic (A3C)

            self.state_return = tf.placeholder(tf.float32, [None])
            self.action = tf.placeholder(tf.float32, [None, board_size * board_size])

            log_probability = tf.log(tf.reduce_sum(tf.mul(self.policy_network, self.action), reduction_indices=1))
            policy_reward = log_probability * (self.state_return - self.value_network) # multiply by the advantage

            value_cost = tf.reduce_mean(tf.square(self.state_return - self.value_network))

            self.train_policy = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=0.99).minimize(-policy_reward) # maximize reward
            self.train_value = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=0.99).minimize(value_cost) # minimize cost of wrong value estimations

    def load_from_checkpoint(self, tfsession):
            # Load network from checkpoint if available
            self.saver = tf.train.Saver()
            tfsession.run(tf.initialize_all_variables())
            checkpoint = tf.train.get_checkpoint_state('saved_networks')
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(tfsession, checkpoint.model_checkpoint_path)
                print('Succesfully loaded network from ' + str(checkpoint.model_checkpoint_path))
            else:
                print('Unable to find any saved networks')

    def policy_output(self, state, tfsession):
        return tfsession.run(self.policy_network, feed_dict={self.state_input: state})

    def value_output(self, state, tfsession):
        return tfsession.run(self.value_network, feed_dict={self.state_input: state})

    def train(self, states, actions, state_returns, tfsession):
        tfsession.run(self.train_policy, feed_dict={self.state_input: states, self.action: actions, self.state_return: state_returns})
        tfsession.run(self.train_value, feed_dict={self.state_input: states, self.state_return: state_returns})

    def save(self, tfsession):
        path = 'saved_networks/Hex9x9-v0-Hexitricty'
        self.saver.save(tfsession, path)
        print('Saved network in ' + path + ' [' + datetime.now().time().isoformat() + ']')
