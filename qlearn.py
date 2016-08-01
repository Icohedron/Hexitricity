import tensorflow as tf
import numpy as np
import gym
import random
import pickle
from tqdm import tqdm

from network import Network

EPISODES = 20000
TIMESTEPS = 1000

EPSILON = 0.5
EPSILON_DECAY = 0.00005
EPSILON_MIN = 0.1

GAMMA = 1.0

REPLAY_SAMPLES = 32

def choose_action(state, tfsession):
    global EPSILON
    if random.random() < EPSILON:
        if EPSILON > EPSILON_MIN:
            EPSILON -= EPSILON_DECAY
        return np.random.choice(np.nonzero(np.array(state[2]).flatten())[0])
    else:
        return network.choose_best_action([state], tfsession)[0]

def sample_replays(batch_size, replay_memory):
    if batch_size > len(replay_memory):
        batch_size = len(replay_memory)
    return random.sample(replay_memory, batch_size)

with tf.Session() as tfsession:

    env = gym.make('Hex9x9-v0')
    network = Network(9, tfsession)
    try:
        replay_memory = pickle.load(open('saved_replays', 'rb'))
    except:
        replay_memory = []

    try:
        for episode in tqdm(range(EPISODES)):

            state = env.reset()

            for timestep in range(TIMESTEPS):

                action = choose_action(state, tfsession)
                state_prime, reward, terminal, info = env.step(action)

                action_one_hot = np.zeros(81)
                action_one_hot[action] = 1

                replay_memory.append((state, action_one_hot, reward, state_prime, terminal))

                replay_batch = sample_replays(REPLAY_SAMPLES, replay_memory)
                replay_states = [play[0] for play in replay_batch]
                replay_actions = [play[1] for play in replay_batch]
                replay_rewards = [play[2] for play in replay_batch]
                replay_states_prime = [play[3] for play in replay_batch]

                true_rewards = []
                Q_batch = network.choose_best_action(replay_states_prime, tfsession)
                for i in range(len(replay_batch)):
                    replay_terminal_state = replay_batch[i][4]
                    if replay_terminal_state:
                        true_rewards.append(replay_rewards[i])
                    else:
                        true_rewards.append(replay_rewards[i] + GAMMA * Q_batch[i])

                network.train(true_rewards, replay_states, replay_actions)

                state = state_prime

                if terminal:
                    # print(action)
                    print('Finished episode {0} in {1} timesteps.'.format(episode, timestep))
                    break

    except KeyboardInterrupt:
        print('Stopping training...')

    print('Saving network...')
    network.save(tfsession)
    pickle.dump(replay_memory, open('saved_replays', 'wb'))
