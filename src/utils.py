import datetime as dt
import math
import logging
import pandas as pd
import numpy as np
import pdb
from statsmodels.tsa.arima_model import ARIMA

def timestamp():
  return round(dt.datetime.now().timestamp())

def format_position(price):
  if price < 0:
    return f'-${abs(price)}'
  else:
    return f'+${abs(price)}'

def format_currency(price):
  return f'${abs(price)}'

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def get_state(data, t, n_days, mode=None):
  res = []
  d = t - n_days + 1

  if d >= 0:
    block = data[d: t + 1]
  else:
   block = -d * [data[0]] + data[0: t + 1]


  if mode == 'arima' and d >= 0:
    model = ARIMA(block, order = (5, 1, 0))
    model_fit = model.fit(disp = 0)
    block, _, _ = model_fit.forecast(steps = len(block))

  for i in range(n_days - 1):
      res.append(sigmoid(block[i + 1] - block[i]))
  return np.array([res])

def show_training_result(result, val_position, initial_offset):
  if val_position != initial_offset and val_position != 0.0:
    logging.info(f'Episode {result[0]}/{result[1]} - Train Position: {format_position(result[2])}  Val Position: {format_position(val_position)}  Train Loss: {result[3]})')


def show_evaluation_result(profit, initial_offset):
  if profit != initial_offset and profit != 0.0:
    logging.info(f'{format_position(profit)}\n')

def get_stock_data(stock_file):
  df = pd.read_csv(stock_file)
  return list(df['Adj Close'])


