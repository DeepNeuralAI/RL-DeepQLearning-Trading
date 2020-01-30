import os
import logging
import numpy as np
from tqdm import tqdm
from src.utils import get_state, format_currency, format_position
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

def train_model(agent, episode, data, episode_count = 50, batch_size = 32, window_size = 10):
  total_profit = 0
  num_observations = len(data) - 1

  agent.inventory = []
  shares = []
  average_loss = []

  state = get_state(data, 0, window_size + 1)

  for t in tqdm(range(num_observations), total = num_observations, leave = True, desc = f'Episode {episode}/{episode_count}'):
    reward = 0
    next_state = get_state(data, t + 1, window_size + 1)
    action = agent.action(state)

    if action == 1:
      agent.inventory.append(data[t])
      shares.append(1)
    elif action == 2 and len(agent.inventory) > 0:
      purchase_price = agent.inventory.pop(0)
      delta = data[t] - purchase_price
      reward = delta
      total_profit += delta
      shares.append(-1)
    else:
      shares.append(0)

    done = (t == (num_observations - 1))
    agent.remember(state, action, reward, next_state, done)

    if len(agent.memory) > batch_size:
      loss = agent.replay(batch_size)
      average_loss.append(loss)

    state = next_state

    if episode % 10 == 0:
      agent.save(episode)

  return (episode, episode_count, total_profit, np.array(average_loss).mean())


def evaluate_model(agent, data, window_size, verbose):
  total_profit = 0
  num_observations = len(data) - 1

  shares = []
  history = []
  agent.inventory = []

  state = get_state(data, 0, window_size + 1)

  for t in range(num_observations):
    reward = 0
    next_state = get_state(data, t + 1, window_size + 1)

    action = agent.action(state, eval = True)

    if action == 1:
      agent.inventory.append(data[t])
      shares.append(1)
      history.append((data[t], "BUY"))

      if verbose:
        logging.debug(f"Buy at: {format_currency(data[t])}")

    elif action == 2 and len(agent.inventory) > 0:
      purchase_price = agent.inventory.pop(0)
      delta = data[t] - purchase_price
      reward = delta
      total_profit += delta
      shares.append(-1)
      history.append((data[t], "SELL"))

      if verbose:
        logging.debug(f"Sell at: {format_currency(data[t])} | Position: {format_position(data[t] - purchase_price)}")

    else:
      history.append((data[t], "HOLD"))
      shares.append(0)

    done = (t == num_observations - 1)
    agent.memory.append((state, action, reward, next_state, done))
    state = next_state

    if done:
      return total_profit, history, shares


