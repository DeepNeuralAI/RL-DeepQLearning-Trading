import os
import logging
import numpy as np
from tqdm import tqdm
from src.utils import get_state, format_currency, format_position, normalize
from tensorboardX import SummaryWriter
import pdb

def train_model(agent, episode, data, episode_count = 50, batch_size = 32, window_size = 10, mode = None):
  total_profit = 0
  num_observations = len(data)

  agent.inventory = []
  shares = []
  average_loss = []
  normed_data = normalize(data)

  for t in tqdm(range(num_observations), total = num_observations, leave = True, desc = f'Episode {episode}/{episode_count}'):
    reward = 0
    done = t == (num_observations - 1)

    state = get_state(normed_data, t)
    action = agent.action(state)

    if action == 1: # Buy
      agent.inventory.append(data.price[t])
      shares.append(1)
      reward -= 0.1 # Commission Penalty
    elif action == 2 and len(agent.inventory) > 0: # Sell
      purchase_price = agent.inventory.pop(0)
      delta = data.price[t] - purchase_price
      reward = delta - 0.1 # Commission Penalty
      total_profit += delta
      shares.append(-1)
    else: # Hold
      shares.append(0)

    if len(agent.memory) > batch_size:
      loss = agent.replay(batch_size)
      average_loss.append(loss)

    if not done:
      next_state = get_state(normed_data, t + 1)
      agent.remember(state, action, reward, next_state, done)
      state = next_state

    if episode % 10 == 0:
      agent.save(episode)

    if done:
      return (episode, episode_count, total_profit, np.array(average_loss).mean())

def evaluate_model(agent, data, verbose, window_size = 10):
  total_profit = 0
  num_observations = len(data)

  shares = []
  history = []
  agent.inventory = []
  normed_data = normalize(data)
  cum_return = []

  for t in range(num_observations):
    done = t == (num_observations - 1)
    reward = 0

    state = get_state(normed_data, 0)
    action = agent.action(state, evaluation = True)

    if action == 1:
      agent.inventory.append(data.price[t])
      shares.append(1)
      history.append((data.price[t], "BUY"))

      if verbose:
        logging.debug(f"Buy at: {format_currency(data.price[t])}")

    elif action == 2 and len(agent.inventory) > 0:
      purchase_price = agent.inventory.pop(0)
      delta = data.price[t] - purchase_price
      reward = delta
      total_profit += delta
      shares.append(-1)
      history.append((data.price[t], "SELL"))

      if verbose:
        logging.debug(f"Sell at: {format_currency(data.price[t])} | Position: {format_position(data.price[t] - purchase_price)}")

    else:
      history.append((data.price[t], "HOLD"))
      shares.append(0)
    cum_return.append(total_profit)

    if not done:
      next_state = get_state(normed_data, t + 1)
      agent.memory.append((state, action, reward, next_state, done))
      state = next_state

    if done:
      return total_profit, history, shares, cum_return


