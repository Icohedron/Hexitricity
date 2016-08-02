import tensorflow as tf
import math
from datetime import datetime

WEIGHT_STDDEV = 0.01
BIAS_CONSTANT = 0.01

RMSProp_DECAY = 0.99
LEARNING_RATE = 1e-4

NETWORK_SAVE_PATH = 'saved_networks'
NETWORK_SAVE_NAME = 'Hex9x9-v0-Hexitricty.checkpoint'
FULL_NETWORK_PATH = NETWORK_SAVE_PATH + '/' + NETWORK_SAVE_NAME

SUMMARY_FILE_PATH = '/tmp/a3c_hex/tf_summaries'


def relu_conv_layer(input_img, kernel_size, in_channels, out_channels):
    W = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, in_channels, out_channels], stddev=WEIGHT_STDDEV), name='weights')
    b = tf.Variable(tf.constant(BIAS_CONSTANT, shape=[out_channels]), name='bias')
    conv = tf.nn.conv2d(input_img, W, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + b)


def create_network(graph, board_size, thread_sub_network=False):
    with graph.as_default():
        with graph.device('/cpu:0'):

            with tf.name_scope('Inputs'):
                state = tf.placeholder(tf.float32, shape=[None, 3, board_size, board_size], name='state')
                action = tf.placeholder(tf.int32, shape=[None], name='action')
                reward = tf.placeholder(tf.float32, shape=[None], name='reward')

                state_img = tf.transpose(state, perm=[0, 2, 3, 1])

            with tf.name_scope('HiddenConv-L0'):
                h_conv0 = relu_conv_layer(state_img, 5, 3, 30)

            with tf.name_scope('HiddenConv-L1'):
                h_conv1 = relu_conv_layer(h_conv0, 4, 30, 60)

            with tf.name_scope('HiddenConv-L2'):
                h_conv2 = relu_conv_layer(h_conv1, 3, 60, 120)

                conv_flat = tf.reshape(h_conv2, [-1, board_size * board_size * 120])

            with tf.name_scope('HiddenFC-L3'):
                W_fcl = tf.Variable(tf.truncated_normal([board_size * board_size * 120, 512], stddev=WEIGHT_STDDEV), name='weights')
                b_fcl = tf.Variable(tf.constant(BIAS_CONSTANT, shape=[512]), name='bias')

                h_fcl = tf.nn.relu(tf.matmul(conv_flat, W_fcl) + b_fcl)

            with tf.name_scope('OutputPolicy-L4-A'):
                p_W_fcl = tf.Variable(tf.truncated_normal([512, board_size * board_size], stddev=WEIGHT_STDDEV), name='weights')
                p_b_fcl = tf.Variable(tf.constant(BIAS_CONSTANT, shape=[board_size * board_size]), name='bias')

                policy = tf.nn.softmax(tf.matmul(h_fcl, p_W_fcl) + p_b_fcl)

            with tf.name_scope('OutputValue-L4-B'):
                v_W_fcl = tf.Variable(tf.truncated_normal([512, 1], stddev=WEIGHT_STDDEV), name='weights')
                v_b_fcl = tf.Variable(tf.constant(BIAS_CONSTANT, shape=[1]), name='bias')

                value = tf.matmul(h_fcl, v_W_fcl) + v_b_fcl

            # Training

            optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=RMSProp_DECAY)

            action_one_hot = tf.one_hot(action, board_size * board_size)
            reward_expanded = tf.expand_dims(reward, -1)

            # policy_debug = tf.Print(policy, [policy], 'Policy: ')
            # value_debug = tf.Print(value, [value], 'Value: ')
            # reward_expanded_debug = tf.Print(reward_expanded, [reward_expanded], 'Reward: ')

            # policy_baseline = tf.log(tf.reduce_sum(tf.mul(policy_debug, action_one_hot), reduction_indices=1)) * (reward_expanded_debug - value_debug)
            policy_baseline = tf.log(tf.reduce_sum(tf.mul(policy, action_one_hot), reduction_indices=1)) * (reward_expanded - value)
            value_loss = tf.reduce_mean(tf.square(reward - value))

            # policy_baseline_debug = tf.Print(policy_baseline, [policy_baseline], 'Policy Baseline: ')
            # value_loss_debug = tf.Print(value_loss, [value_loss], 'Value Loss: ')

            # policy_optimizer = optimizer.minimize(-policy_baseline_debug) # minimizing a negative is the same as maximizing a positive
            policy_optimizer = optimizer.minimize(-policy_baseline) # minimizing a negative is the same as maximizing a positive
            # value_optimizer = optimizer.minimize(value_loss_debug)
            value_optimizer = optimizer.minimize(value_loss)

            # Add variables and operations to graph

            tf.add_to_collection('inputs', state)
            tf.add_to_collection('inputs', action)
            tf.add_to_collection('inputs', reward)

            tf.add_to_collection('outputs', policy)
            tf.add_to_collection('outputs', value)

            tf.add_to_collection('optimizers', policy_optimizer)
            tf.add_to_collection('optimizers', value_optimizer)

            tf.add_to_collection('test', v_b_fcl)

            tf.add_to_collection('initializer', tf.initialize_all_variables())


def save_network(saver, session):
    saver.save(session, FULL_NETWORK_PATH)
    print('Saved network in ' + FULL_NETWORK_PATH + ' [' + datetime.now().time().isoformat() + ']')


def restore_checkpoint(saver, session):
    checkpoint = tf.train.get_checkpoint_state(NETWORK_SAVE_PATH)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(session, checkpoint.model_checkpoint_path)
        print('Restored session from previous checkpoint.')
    else:
        print('Unable to find a saved checkpoint.\nStarting a new network.')
        save_network(saver, session)
