# Based on https://github.com/coreylynch/async-rl/blob/master/a3c.py -- last commit: dd00edc on June 24, 2016

import tensorflow as tf
import numpy as np
import gym
import threading
import random
import time
from tqdm import tqdm, trange

from network import Shared_VP_Network

TRAIN = True # else, EVALUATE

CONCURRENT_THREADS = 4

REWARD_DISCOUNT_GAMMA = 0.99

BOARD_SIZE = 9
BOARD_SIZE_SQUARED = BOARD_SIZE ** 2

SAVE_INTERVAL = 5000
SUMMARY_INTERVAL = 5

SUMMARY_FILE_PATH = '/tmp/a3c_hex/tf_summaries'
EVALULATION_FILE_PATH = '/tmp/a3c_hex/eval'

T = 0
T_max = 900000000

t_max = 42


def a3c_thread(thread_num, keep_running, environment, network, summary_ops, tfsession):
    # Algorithm S3 from https://arxiv.org/abs/1602.01783
    # Google Deepmind -- Asynchronous Methods for Deep Reinforcement Learning
    # Asynchronous Advantage Actor Critic (A3C)

    global T, T_max
    time.sleep(3 * thread_num)

    reward_summary_placeholder, update_episode_reward, summary_op = summary_ops

    while T < T_max and keep_running.is_set():
        states_batch = []
        actions_batch = []
        rewards_batch = []

        t = 0
        t_start = t
        state = environment.reset()
        terminal = False

        episode_reward = 0

        while not terminal or (t - t_start) == t_max:
            action_probability_distribution = network.policy_output([state], tfsession)[0]
            action_index = np.random.choice(len(action_probability_distribution), p=action_probability_distribution)

            action_one_hot = np.zeros(BOARD_SIZE_SQUARED)
            action_one_hot[action_index] = 1

            next_state, reward, terminal, info = environment.step(action_index)

            states_batch.append(state)
            actions_batch.append(action_one_hot)
            rewards_batch.append(reward)

            episode_reward += reward

            t += 1
            T += 1

            state = next_state

        if terminal:
            state_return = 0
        else:
            state_return = network.value_output([state])[0][0] # Bootstrap from last state

        state_returns_batch = np.zeros(t)
        for i in reversed(range(t_start, t)):
            state_return = rewards_batch[i] + REWARD_DISCOUNT_GAMMA * state_return
            state_returns_batch[i] = state_return

        network.train(states_batch, actions_batch, state_returns_batch, tfsession)

        if terminal:
            # Record statistics of finished episode
            # print('Thread {0} finished episode in {1} timesteps with reward {2} at time {3}'.format(thread_num, t, episode_reward, T))
            print(action_index)
            tfsession.run(update_episode_reward, feed_dict={reward_summary_placeholder: episode_reward})


def get_summary_ops():
    # Generate summary operations for Tensorboard

    episode_reward = tf.Variable(0.0)
    tf.scalar_summary('Episode Reward', episode_reward)
    reward_summary_placeholder = tf.placeholder(tf.float32)
    update_episode_reward = episode_reward.assign(reward_summary_placeholder)

    summary_op = tf.merge_all_summaries()

    return reward_summary_placeholder, update_episode_reward, summary_op


def train(network, tfsession):
    environments = [gym.make('Hex9x9-v0') for i in range(CONCURRENT_THREADS)]

    keep_running = threading.Event()
    keep_running.set()

    summary_ops = get_summary_ops()
    summary_op = summary_ops[-1]

    network.load_from_checkpoint(tfsession)
    writer = tf.train.SummaryWriter(SUMMARY_FILE_PATH, tfsession.graph)

    a3c_threads = [threading.Thread(target=a3c_thread, args=(thread_num, keep_running, environments[thread_num], network, summary_ops, tfsession)) for thread_num in range(CONCURRENT_THREADS)]
    for thread in a3c_threads:
        thread.start()

    try:

        last_summary_time = 0
        while True:
            if T % SAVE_INTERVAL == 0 and T != 0:
                network.save(tfsession)

            time_now = time.time()
            if time_now - last_summary_time > SUMMARY_INTERVAL:
                summary = tfsession.run(summary_op)
                writer.add_summary(summary, float(T))
                last_summary_time = time_now

    except KeyboardInterrupt:
        print('Closing threads...')
        keep_running.clear()
        for thread in a3c_threads:
            thread.join()
        print('Succesfully closed all threads.')


def evaluate(network, tfsession):
    environment = gym.make('Hex9x9-v0')
    environment.monitor.start(EVALULATION_FILE_PATH)

    for episode in trange(100):
        state = environment.reset()
        terminal = False

        episode_reward = 0

        while not terminal:
            environment.render()

            action_probability_distribution = network.policy_output([state], tfsession)[0]
            action_index = np.random.choice(len(action_probability_distribution), p=action_probability_distribution)

            state, reward, terminal, info = environment.step(action_index)

            episode_reward += reward

    environment.monitor.close()


with tf.Session() as tfsession:
    network = Shared_VP_Network(BOARD_SIZE, tfsession)
    if TRAIN:
        train(network, tfsession)
    else:
        evaluate(network, tfsession)
