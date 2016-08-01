import tensorflow as tf
import math

# WEIGHT_STDDEV = 0.01
BIAS_CONSTANT = 0.01
LEARNING_RATE = 1e-6

def weight_variable(shape, num_inputs):
    return tf.Variable(tf.truncated_normal(shape, stddev=(1.0 / math.sqrt(num_inputs))), name='weights')

def bias_variable(shape):
    return tf.Variable(tf.constant(BIAS_CONSTANT, shape=shape), name='bias')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def relu_convolutional_layer(x, W, layer_num):
    with tf.name_scope('HiddenConv-L{}'.format(layer_num)):
        W_conv = weight_variable(W, W[-2])
        b_conv = bias_variable([W[-1]])
    return tf.nn.relu(conv2d(x, W_conv) + b_conv)

class Network(object):

    def __init__(self, board_size, tfsession):
        self.__create_network(board_size)
        self.__create_training_methods(board_size)

        # Load network from checkpoint if available
        self.saver = tf.train.Saver()
        tfsession.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state('saved_networks')
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(tfsession, checkpoint.model_checkpoint_path)
            print('Succesfully loaded network from ' + str(checkpoint.model_checkpoint_path))
        else:
            print('Unable to find any saved networks')

    def __create_network(self, board_size):
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

        with tf.name_scope('OutputFC-L5'):
            W_fcl5 = weight_variable([512, board_size * board_size], 512)
            b_fcl5 = bias_variable([board_size * board_size])

        self.Q_value_array = tf.matmul(h_fcl4, W_fcl5) + b_fcl5
        self.Q_best_action = tf.argmax(self.Q_value_array, 1)

    def __create_training_methods(self, board_size):
        self.action_input = tf.placeholder(tf.float32, shape=[None, board_size * board_size])
        self.true_reward = tf.placeholder(tf.float32, shape=[None])
        Q_action = tf.reduce_sum(tf.mul(self.Q_value_array, self.action_input), reduction_indices=1)
        self.cost_function = tf.reduce_mean(tf.square(self.true_reward - Q_action))
        self.train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost_function)

    def get_Q_value_array(self, state_input, tfsession):
        return tfsession.run(self.Q_value_array, feed_dict={self.state_input: state_input})

    def choose_best_action(self, state_input, tfsession):
        return tfsession.run(self.Q_best_action, feed_dict={self.state_input: state_input})

    def train(self, true_reward, state_input, action_input):
        self.train_step.run(feed_dict={self.true_reward: true_reward, self.state_input: state_input, self.action_input: action_input})

    def save(self, tfsession):
        self.saver.save(tfsession, 'saved_networks/Hex9x9-v0-Hexitricty')
        print('Saved network in ' + 'saved_networks/Hex9x9-v0-Hexitricty')
