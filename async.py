# Based on coreylynch's code
# https://github.com/coreylynch/async-rl/blob/master/a3c.py
# Last commit: dd00edc

import tensorflow as tf
import numpy as np
import gym
import threading
import time
import random
from tqdm import tqdm

from network import *

# Adjustable values

TRAIN = True # else, EVALUATE

CONCURRENT_THREADS = 16

EPSILON_EXPLORATION = 1.0
EPSILON_EXPLORATION_DECAY = 5e-5
EPSILON_EXPLORATION_MIN = 0.1

THREAD_START_DELAY = 1

REWARD_DISCOUNT_GAMMA = 1.0

SAVE_INTERVAL = 5000
SUMMARY_INTERVAL = 5

SEED = 64

# Constants

SUMMARY_FILE_PATH = 'saved_networks/tf_summaries'
EVALULATION_FILE_PATH = 'saved_networks/evaluations'

BOARD_SIZE = 9
BOARD_SIZE_SQUARED = BOARD_SIZE ** 2

MAX_T_PER_EPISODE = int(BOARD_SIZE * BOARD_SIZE * 0.5)

# Global variables

T = 0
T_max = 1000000000

t_max = 5


def choose_action(empty_tiles, action_policy, epsilon):
    ap = np.array(action_policy)
    ap *= np.array(empty_tiles).flatten()
    ap /= np.sum(ap)
    if random.random() < epsilon:
        return np.random.choice(len(ap), p=ap)
    else:
        return np.argmax(ap)


def a3c_thread(thread_num, environment, graph, session, summary_ops, thread_coordinator):
    # Algorithm S3 from https://arxiv.org/abs/1602.01783
    # Google Deepmind -- Asynchronous Methods for Deep Reinforcement Learning
    # Asynchronous Advantage Actor Critic (A3C)

    global T, T_max, running

    time.sleep(THREAD_START_DELAY * thread_num)

    print('A3C thread {} started.'.format(thread_num))

    with thread_coordinator.stop_on_exception():

        n_state, n_action, n_reward, n_temporal_difference = graph.get_collection('inputs')
        n_policy, n_value = graph.get_collection('outputs')
        n_optimizer = graph.get_collection('optimizer')[0]

        episode_reward_summary, episode_timesteps_summary, max_action_probability_summary = summary_ops
        
        epsilon = EPSILON_EXPLORATION

        ep_r = 0
        max_ap = 0
        ep_t = 0

        ep_c = 0
        ep_accum_r = []

        state = environment.reset()

        while T < T_max and not thread_coordinator.should_stop():

            states = []
            actions = []
            rewards = []
            temporal_differences = []

            t = 0
            t_start = t
            terminal = False

            while not terminal or (t - t_start) == t_max:
                action_policy = session.run(n_policy, feed_dict={n_state: [state]})[0]
                action = choose_action(state[2], action_policy, epsilon)

                if epsilon > EPSILON_EXPLORATION_MIN:
                    epsilon -= EPSILON_EXPLORATION_DECAY

                next_state, reward, terminal, info = environment.step(action)

                temporal_differences.append(reward - session.run(n_value, feed_dict={n_state: [state]})[0][0])

                ep_t += 1
                if ep_t == MAX_T_PER_EPISODE:
                    terminal = True

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                ep_r += reward

                max_ap_t = np.max(action_policy)
                if max_ap_t > max_ap:
                    max_ap = max_ap_t

                t += 1
                T += 1

                state = next_state

            if terminal:
                state_reward = 0
            else:
                state_reward = session.run(n_value, feed_dict={n_state: [state]})[0][0] # Bootstrap from last state

            state_rewards = np.zeros(t)
            for i in reversed(range(t_start, t)):
                state_reward = rewards[i] + REWARD_DISCOUNT_GAMMA * state_reward
                state_rewards[i] = state_reward

            session.run(n_optimizer, feed_dict={n_state: states, n_action: actions, n_reward: state_rewards, n_temporal_difference: temporal_differences})

            if terminal:
                session.run(episode_timesteps_summary[0], feed_dict={episode_timesteps_summary[1]: ep_t})
                session.run(max_action_probability_summary[0], feed_dict={max_action_probability_summary[1]: max_ap})

                ep_c += 1
                ep_accum_r.append(ep_r)

                if ep_c % 10 == 0:
                    session.run(episode_reward_summary[0], feed_dict={episode_reward_summary[1]: np.mean(ep_accum_r)})
                    ep_accum_r = []

                ep_r = 0
                max_ap = 0
                ep_t = 0
                state = environment.reset()

    print('Closed A3C thread {}.'.format(thread_num))


def gen_summary_ops():
    episode_reward = tf.Variable(0.0, name='episode_reward')
    tf.scalar_summary('Reward Averages over 10 Episodes', episode_reward)
    episode_reward_placeholder = tf.placeholder(tf.float32, name='episode_reward_placeholder')
    update_episode_reward = episode_reward.assign(episode_reward_placeholder)

    episode_timesteps = tf.Variable(0, name='episode_timesteps')
    tf.scalar_summary('Episode Timesteps', episode_timesteps)
    episode_timesteps_placeholder = tf.placeholder(tf.int32, name='episode_timesteps_placeholder')
    update_episode_timesteps = episode_timesteps.assign(episode_timesteps_placeholder)

    max_action_probability = tf.Variable(0.0, name='max_action_probability')
    tf.scalar_summary('Max Action Probability', max_action_probability)
    max_action_probability_placeholder = tf.placeholder(tf.float32, name='max_action_probability_placeholder')
    update_max_action_probability = max_action_probability.assign(max_action_probability_placeholder)

    return (update_episode_reward, episode_reward_placeholder),\
           (update_episode_timesteps, episode_timesteps_placeholder),\
           (update_max_action_probability, max_action_probability_placeholder)


def train(graph, saver, session):
    global T

    environments = [gym.make('Hex9x9-v0') for i in range(CONCURRENT_THREADS)]

    thread_coordinator = tf.train.Coordinator()

    summary_ops = gen_summary_ops()
    summary_writer = tf.train.SummaryWriter(SUMMARY_FILE_PATH, graph)

    session.run(tf.initialize_all_variables())

    a3c_threads = [threading.Thread(target=a3c_thread, args=(thread_num, environments[thread_num], graph, session, summary_ops, thread_coordinator)) for thread_num in range(CONCURRENT_THREADS)]
    print('Starting A3C threads...')
    for thread in a3c_threads:
        thread.start()

    try:

        time.sleep(THREAD_START_DELAY * CONCURRENT_THREADS)

        with tqdm(total=T_max) as tqdmT:
            last_summary_time = time.time()
            last_T = 0

            while True:

                if T % SAVE_INTERVAL == 0:
                    save_network(saver, session)

                time_now = time.time()
                if time_now - last_summary_time > SUMMARY_INTERVAL:
                    summary_string = session.run(tf.merge_all_summaries())
                    summary_writer.add_summary(summary_string, float(T))
                    last_summary_time = time_now

                T_diff = T - last_T
                if T_diff > 0:
                    tqdmT.update(T_diff)
                    last_T = T

    except:
        print('Closing threads...')
        thread_coordinator.request_stop()
        thread_coordinator.join(a3c_threads)
        print('Succesfully closed all threads.')
        save_network(saver, session)


def evaluate(graph, session):
    environment = gym.make('Hex9x9-v0')
    environment.monitor.start(EVALULATION_FILE_PATH)

    n_state, n_action, n_reward = graph.get_collection('inputs')
    n_policy, n_value = graph.get_collection('outputs')

    episode_rewards = []

    session.run(tf.initialize_all_variables())

    for episode in range(100):
        state = environment.reset()
        terminal = False

        ep_t = 0

        while not terminal:
            action_policy = session.run(n_policy, feed_dict={n_state: [state]})[0]
            action = choose_action(state[2], action_policy, 0.0)

            state, reward, terminal, info = environment.step(action)

            ep_t += 1
            if ep_t == MAX_T_PER_EPISODE:
                terminal = True

            episode_rewards.append(reward)

    environment.monitor.close()

    print('Games won (out of 100): {}'.format(len(episode_rewards) - np.count_nonzero(np.array(episode_rewards) - 1.0)))
    print('Games lost (out of 100): {}'.format(len(episode_rewards) - np.count_nonzero(np.array(episode_rewards) + 1.0)))


graph = tf.Graph()
with tf.Session(graph=graph) as session:
    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    
    create_network(graph, BOARD_SIZE)
    saver = tf.train.Saver()

    session.run(tf.initialize_all_variables())
    restore_checkpoint(saver, session)

    if TRAIN:
        train(graph, saver, session)
    else:
        evaluate(graph, session)
