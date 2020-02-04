import os
import logging
import numpy as np
from tqdm import tqdm
from src.utils import get_state, format_currency, format_position, normalize
import pdb

'''
1. Move daily_pct_return to utils
2. Move calc_reward to utils
'''

def daily_pct_change(prices, shift):
  pct_change = (prices.copy() / prices.copy().shift(periods = shift)) - 1
  pct_change[:shift] = 0
  return pct_change

def calc_reward(pct_change, net_holdings):
  return pct_change * net_holdings

def train_model(agent, episode, data, episode_count = 50, batch_size = 32, window_size = 10):
  total_profit = 0
  num_observations = len(data)

  agent.inventory = []
  shares_history = []
  average_loss = []

  net_holdings = 0
  normed_data = normalize(data)
  pct_change = daily_pct_change(data.price, window_size)

  for t in tqdm(range(num_observations), total = num_observations, leave = True, desc = f'Episode {episode}/{episode_count}'):
    done = t == (num_observations - 1)

    state = get_state(normed_data, t)
    action = agent.action(state)

    if action == 2 and net_holdings == 0:
        shares = -10
        net_holdings += -10
    elif action == 2 and net_holdings == 10:
        shares = -20
        net_holdings += -20
    elif action == 1 and net_holdings == 0:
        shares = 10
        net_holdings += 10
    elif action == 1 and net_holdings == -10:
        shares = 20
        net_holdings += 20
    else:
        shares = 0
    shares_history.append(shares)

    reward = calc_reward(pct_change[t], net_holdings)
    total_profit += reward

    # if action == 1: # Buy
    #   agent.inventory.append(data.price[t])
    #
    #   reward -= 1e-5 # Commission Penalty

    # elif action == 2 and len(agent.inventory) > 0: # Sell
    #   purchase_price = agent.inventory.pop(0)
    #   delta = data.price[t] - purchase_price
    #   reward = delta - 1e-5 # Commission Penalty
    #   total_profit += delta
    #   shares.append(-1)

    # else: # Hold
    #   shares.append(0)
    #   reward -= 1e-3

    if not done:
      next_state = get_state(normed_data, t + 1)
      agent.remember(state, action, reward, next_state, done)
      state = next_state

    if len(agent.memory) > batch_size:
      loss = agent.replay(batch_size)
      average_loss.append(loss)

    if episode % 50 == 0:
      agent.save(episode)

    if done: return (episode, episode_count, total_profit, np.array(average_loss).mean())

def evaluate_model(agent, data, verbose, window_size = 10):
  total_profit = 0
  num_observations = len(data)

  shares = []
  history = []
  agent.inventory = []
  normed_data = normalize(data)
  cum_return = []
  net_holdings = 0
  shares_history = []
  pct_change = daily_pct_change(data.price, 10)

  for t in range(num_observations):
    done = t == (num_observations - 1)
    reward = 0

    state = get_state(normed_data, t)
    action = agent.action(state, evaluation = True)

    if action == 2 and net_holdings == 0:
      shares = -10
      net_holdings += -10
      history.append((data.price[t], "SELL"))
    elif action == 2 and net_holdings == 10:
      shares = -20
      net_holdings += -20
      history.append((data.price[t], "SELL"))
    elif action == 1 and net_holdings == 0:
      shares = 10
      net_holdings += 10
      history.append((data.price[t], "BUY"))
    elif action == 1 and net_holdings == -10:
      shares = 20
      net_holdings += 20
      history.append((data.price[t], "BUY"))
    else:
      shares = 0
      history.append((data.price[t], "HOLD"))
    shares_history.append(shares)

    reward = calc_reward(pct_change[t], net_holdings)
    total_profit += reward
    # if action == 1:
    #   agent.inventory.append(data.price[t])
    #   shares.append(1)
    #   history.append((data.price[t], "BUY"))

    #   if verbose:
    #     logging.debug(f"Buy at: {format_currency(data.price[t])}")

    # elif action == 2 and len(agent.inventory) > 0:
    #   purchase_price = agent.inventory.pop(0)
    #   delta = data.price[t] - purchase_price
    #   reward = delta
    #   total_profit += delta
    #   shares.append(-1)
    #   history.append((data.price[t], "SELL"))

    #   if verbose:
    #     logging.debug(f"Sell at: {format_currency(data.price[t])} | Position: {format_position(data.price[t] - purchase_price)}")

    # else:
    #   history.append((data.price[t], "HOLD"))
    #   shares.append(0)
    # cum_return.append(total_profit)

    if not done:
      next_state = get_state(normed_data, t + 1)
      agent.memory.append((state, action, reward, next_state, done))
      state = next_state

    if done: return total_profit, history, shares_history


