import numpy as np
import gym
import random
import pickle
from tqdm import tqdm

EPISODES = 100000
TIMESTEPS = 1000

env = gym.make('Hex9x9-v0')
replay_memory = []

try:
    for episode in tqdm(range(EPISODES)):

        state = env.reset()

        for timestep in range(TIMESTEPS):

            action = np.random.choice(np.nonzero(np.array(state[2]).flatten())[0])
            state_prime, reward, terminal, info = env.step(action)

            action_one_hot = np.zeros(81)
            action_one_hot[action] = 1

            replay_memory.append((state, action_one_hot, reward, state_prime, terminal))

            state = state_prime

            if terminal:
                break

except KeyboardInterrupt:
    print('Stopping...')

pickle.dump(replay_memory, open('saved_replays', 'wb'))
