import tensorflow as tf
import numpy as np
import math
# import pickle
from datetime import datetime

LEARNING_RATE = 7e-4
ENTROPY_REGULARIZATION_FACTOR = 0.01

NETWORK_SAVE_PATH = 'saved_networks'
NETWORK_SAVE_NAME = 'Hex9x9-v0-Hexitricty.checkpoint'
FULL_NETWORK_PATH = NETWORK_SAVE_PATH + '/' + NETWORK_SAVE_NAME

# T = 0

# Weight and bias initialization from miyosuda's code
# https://github.com/miyosuda/async_deep_reinforce/blob/master/game_ac_network.py
# Last commit: 41f2d75

def fc_weight_variable(shape):
    input_channels = shape[0]
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial)


def fc_bias_variable(shape, input_channels):
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial)


def conv_weight_variable(shape):
    w = shape[0]
    h = shape[1]
    input_channels = shape[2]
    d = 1.0 / np.sqrt(input_channels * w * h)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial)


def conv_bias_variable(shape, w, h, input_channels):
    d = 1.0 / np.sqrt(input_channels * w * h)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial)


def relu_conv_layer(input_img, kernel_size, in_channels, out_channels):
    W = conv_weight_variable([kernel_size, kernel_size, in_channels, out_channels])
    b = conv_bias_variable([out_channels], kernel_size, kernel_size, in_channels)
    conv = tf.nn.conv2d(input_img, W, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + b)


def create_network(graph, board_size, thread_sub_network=False):
    with graph.as_default():
        with graph.device('/cpu:0'):

            with tf.name_scope('Inputs'):
                state = tf.placeholder(tf.float32,
                            shape=[None, 3, board_size, board_size],
                            name='state')
                action = tf.placeholder(tf.int32, shape=[None], name='action')
                reward = tf.placeholder(tf.float32, shape=[None], name='reward')
                temporal_difference = tf.placeholder(tf.float32, shape=[None], name='temporal_difference')

                state_img = tf.transpose(state, perm=[0, 2, 3, 1])

            with tf.name_scope('HiddenConv-L0'):
                h_conv0 = relu_conv_layer(state_img, 5, 3, 30)

            with tf.name_scope('HiddenConv-L1'):
                h_conv1 = relu_conv_layer(h_conv0, 4, 30, 60)

            with tf.name_scope('HiddenConv-L2'):
                h_conv2 = relu_conv_layer(h_conv1, 3, 60, 120)

                conv_flat = tf.reshape(h_conv2, [-1, board_size * board_size * 120])

            with tf.name_scope('HiddenFC-L3'):
                W_fcl = fc_weight_variable([board_size * board_size * 120, 512])
                b_fcl = fc_bias_variable([512], board_size * board_size * 120)

                h_fcl = tf.nn.relu(tf.matmul(conv_flat, W_fcl) + b_fcl)

            with tf.name_scope('OutputPolicy-L4-A'):
                p_W_fcl = fc_weight_variable([512, board_size * board_size])
                p_b_fcl = fc_bias_variable([board_size * board_size], 512)

                policy = tf.nn.softmax(tf.matmul(h_fcl, p_W_fcl) + p_b_fcl)

            with tf.name_scope('OutputValue-L4-B'):
                v_W_fcl = fc_weight_variable([512, 1])
                v_b_fcl = fc_bias_variable([1], 512)

                value = tf.matmul(h_fcl, v_W_fcl) + v_b_fcl

            # Training

            # Loss functions from miyosuda's code
            # https://github.com/miyosuda/async_deep_reinforce/blob/master/game_ac_network.py
            # Last commit: 41f2d75

            # Adam = tf.train.AdamOptimizer(LEARNING_RATE)
            RMSProp = tf.train.RMSPropOptimizer(LEARNING_RATE)

            action_one_hot = tf.one_hot(action, board_size * board_size)

            log_policy = -tf.log(tf.clip_by_value(policy, 1e-20, 1.0))

            entropy = -tf.reduce_sum(policy * log_policy, reduction_indices=1)

            policy_loss = -tf.reduce_sum(tf.reduce_sum(tf.mul(log_policy, action_one_hot), reduction_indices=1) * temporal_difference + ENTROPY_REGULARIZATION_FACTOR * entropy)
            value_loss = 0.5 * tf.nn.l2_loss(reward - value)

            total_loss = policy_loss + value_loss

            # optimizer = Adam.minimize(total_loss)
            optimizer = RMSProp.minimize(total_loss)

            # Add variables and operations to graph

            tf.add_to_collection('inputs', state)
            tf.add_to_collection('inputs', action)
            tf.add_to_collection('inputs', reward)
            tf.add_to_collection('inputs', temporal_difference)

            tf.add_to_collection('outputs', policy)
            tf.add_to_collection('outputs', value)

            tf.add_to_collection('optimizer', optimizer)


def save_network(saver, session):
    # global T
    saver.save(session, FULL_NETWORK_PATH)
    # pickle.dump(T, open(NETWORK_SAVE_PATH + '/global_step', 'wb'))
    print('Saved network in ' + FULL_NETWORK_PATH + ' [' + datetime.now().time().isoformat() + ']')


def restore_checkpoint(saver, session):
    # global T
    checkpoint = tf.train.get_checkpoint_state(NETWORK_SAVE_PATH)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(session, checkpoint.model_checkpoint_path)
        # T = pickle.load(open(NETWORK_SAVE_PATH + '/global_step', 'rb'))
        print('Restored session from previous checkpoint.')
    else:
        print('Unable to find a saved checkpoint.\nStarting a new network.')
        save_network(saver, session)
