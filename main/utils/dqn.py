import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import sys
import pandas as pd
import numpy as np
import datetime as dt
import pdb

sys.path.append('../')
sys.path.append('../utils/')
sys.path.append('../../data/')

from util import data_df

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, num_observations, num_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_observations, 24),
            nn.ReLU(),
            nn.Linear(24, 48),
            nn.ReLU(),
            nn.Linear(48, num_actions)
        )

    def forward(self, x):
        return self.model(x).double()

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

n_actions = 3
num_observations = 14

policy_net = DQN(14, 3).to(device)

target_net = DQN(14, 3).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(1000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

episode_durations = []
def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)

    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device, dtype=torch.double)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    # Optimize the model
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

class TradingEnv(gym.Env):

    def __init__(self, symbol, start_date, end_date, span = 14):
        self.symbol, self.span = symbol, span
        self.start_date, self.end_date = start_date, end_date
        self.data = self.get_data()
        self.observations = self.data
        self.inventory = []

        self.steps = self.data.shape[0] - 1
        self.observation_space = gym.spaces.Box(low = -np.inf, high = np.inf,
                                                shape = (self.span,), dtype = np.float32)
        self.action_space = gym.spaces.Discrete(3)


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def reset(self):
        self.total_profit = 0.
        self.t = 0
        self.inventory = []

    def get_data(self):
        data = data_df(self.symbol, n_period = self.span)
        date_range = pd.date_range(start = self.start_date, end = self.end_date)
        data_ = data.loc[date_range].dropna(subset=[self.symbol])
        return data_[self.symbol]

    def getState(self, observations, t, lag):
        d = t - lag + 1

        if d >= 0:
            block = observations[d:t+1]
        else:
            bfill = -d * [observations[0]]
            block =  bfill + observations[0:t + 1].values.tolist()

        res = []

        for i in range(lag - 1):
            res.append(self.sigmoid(block[i + 1] - block[i]))
        return torch.FloatTensor([res])

    def step(self, action):
        info = {"symbol": self.symbol, "total_profit": self.total_profit}

        # BUY:
        if action == 1:
            self.inventory.append(self.observations[self.t])
            margin = 0.
            reward = 0.
        # SELL:
        elif action == 2 and len(self.inventory) > 0:
            bought_price = self.inventory.pop(0)
            margin = self.observations[self.t] - bought_price
            reward = max(margin, 0.)
        # HOLD:
        else:
            margin = 0.
            reward = 0.

        self.total_profit += margin

        # Increment time
        self.t += 1

        obs = self.getState(self.observations, self.t, self.span + 1)


        # Stop episode when reaches last time step in file
        if self.t >= self.steps:
            done = True
        else:
            done = False


        return obs, reward, done, info




num_episodes = 2

env = TradingEnv('AAPL', start_date=dt.datetime(2016, 1, 1),
                 end_date = dt.datetime(2016, 12, 31), span = 14)

for i_episode in range(num_episodes):
    print(f'Episode Num: {i_episode}')
    # Initialize the environment and state
    env.reset()
    state = env.getState(env.observations, 0, 14 + 1)

    for t in count():
        print(f'Count: {t}')
        # Select and perform an action
        action = select_action(state)
        obs, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device, dtype=torch.float64)
        print(reward)

        if not done:
            next_state = obs
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)
        print('Memory Pushed')

#       # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        print('Optimized')

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
