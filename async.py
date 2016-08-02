# Based on https://github.com/coreylynch/async-rl/blob/master/a3c.py -- last commit: dd00edc on June 24, 2016

import tensorflow as tf
import numpy as np
import gym
import threading
import time
from tqdm import tqdm

from network import *

TRAIN = True # else, EVALUATE

CONCURRENT_THREADS = 4 # Adjust this according to your CPU specs

THREAD_START_DELAY = 1

REWARD_DISCOUNT_GAMMA = 0.99

BOARD_SIZE = 9
BOARD_SIZE_SQUARED = BOARD_SIZE ** 2

SAVE_INTERVAL = 5000
SUMMARY_INTERVAL = 5

NETWORK_UPDATE_INTERVAL = 42

EVALULATION_FILE_PATH = '/tmp/a3c_hex/eval'

T = 0
T_max = 2000

t_max = 8


def a3c_thread(thread_num, environment, graph, session, thread_coordinator):
    # Algorithm S3 from https://arxiv.org/abs/1602.01783
    # Google Deepmind -- Asynchronous Methods for Deep Reinforcement Learning
    # Asynchronous Advantage Actor Critic (A3C)

    global T, T_max, running

    time.sleep(THREAD_START_DELAY * thread_num)

    print('A3C thread {} started.'.format(thread_num))

    with thread_coordinator.stop_on_exception():

        n_state, n_action, n_reward = graph.get_collection('inputs')
        n_policy, n_value = graph.get_collection('outputs')
        n_policy_optimizer, n_value_optimizer = graph.get_collection('optimizers')

        state = environment.reset()

        while T < T_max and not thread_coordinator.should_stop():

            states = []
            actions = []
            rewards = []

            # print('Thread {0} start loop: {1} at global step {2}'.format(thread_num, session.run(graph.get_collection('test')[0]), T))

            t = 0
            t_start = t
            terminal = False

            while not terminal or (t - t_start) == t_max:
                action_policy = session.run(n_policy, feed_dict={n_state: [state]})[0]
                action = np.random.choice(len(action_policy), p=action_policy)

                next_state, reward, terminal, info = environment.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                t += 1
                T += 1

                state = next_state

            if terminal:
                state_reward = 0
            else:
                state_reward = session.run(n_value, feed_dict={n_state: [state]})[0][0] # Bootstrap from last state

            state_rewards = []
            for i in reversed(range(t_start, t)):
                state_reward = rewards[i] + REWARD_DISCOUNT_GAMMA * state_reward
                state_rewards.append(state_reward)

            states_rev = list(reversed(states))
            acitons_rev = list(reversed(actions))

            session.run(n_policy_optimizer, feed_dict={n_state: states_rev, n_action: acitons_rev, n_reward: state_rewards})
            session.run(n_value_optimizer, feed_dict={n_state: states_rev, n_reward: state_rewards})

            # print('Thread {0} end loop: {1} at global step {2}'.format(thread_num, session.run(graph.get_collection('test')[0]), T))

            if terminal:
                # print(action)
                state = environment.reset()

    print('Closed A3C thread {}.'.format(thread_num))


def train(graph, session):
    environments = [gym.make('Hex9x9-v0') for i in range(CONCURRENT_THREADS)]

    thread_coordinator = tf.train.Coordinator()

    a3c_threads = [threading.Thread(target=a3c_thread, args=(thread_num, environments[thread_num], graph, session, thread_coordinator)) for thread_num in range(CONCURRENT_THREADS)]
    print('Starting A3C threads...')
    for thread in a3c_threads:
        thread.start()

    try:

        time.sleep(THREAD_START_DELAY * CONCURRENT_THREADS)

        with tqdm(total=T_max) as tqdmT:
            last_T = 0
            while True:
                T_diff = T - last_T
                if T_diff > 0:
                    tqdmT.update(T_diff)
                    last_T = T

    except KeyboardInterrupt:
        print('Closing threads...')
        thread_coordinator.request_stop()
        thread_coordinator.join(a3c_threads)
        print('Succesfully closed all threads.')


graph = tf.Graph()
with tf.Session(graph=graph) as session:
    create_network(graph, BOARD_SIZE)
    saver = tf.train.Saver()

    session.run(tf.initialize_all_variables())
    restore_checkpoint(saver, session)

    train(graph, session)
