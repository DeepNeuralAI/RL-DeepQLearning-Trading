import datetime as dt
import math
import logging
import pandas as pd
import numpy as np
import pdb
from statsmodels.tsa.arima_model import ARIMA

from .technical_indicators import (
  indicators_dict,
  add_momentum_indicators,
  add_trend_indicators,
  add_volatility_indicators,
  add_volume_indicators
)

def timestamp():
  return round(dt.datetime.now().timestamp())

def format_position(price):
  if price < 0:
    return f'-${abs(price)}'
  else:
    return f'+${abs(price)}'

def normalize(df):
  result = df.copy()
  for feature_name in df.columns:
    max_value = df[feature_name].max()
    min_value = df[feature_name].min()
    result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
  return result

def format_currency(price):
  return f'${abs(price)}'

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def get_state(data, t):
  return np.array([data.iloc[t]])

def show_training_result(result, val_position):
  logging.info(f'Episode {result[0]}/{result[1]} - Train Position: {format_position(result[2])}  Val Position: {format_position(val_position)}  Train Loss: {result[3]})')

def show_evaluation_result(profit):
  logging.info(f'{format_position(profit)}\n')

def get_stock_data(stock_file):
  df = pd.read_csv(stock_file)
  return list(df['Adj Close'])

def load_data(path):
  temp = pd.read_csv(path)

  if "Date" in temp.columns:
      temp.index = temp["Date"]
      temp.index = pd.to_datetime(temp.index, infer_datetime_format=True)
      temp.drop('Date', axis = 1, inplace = True)
      temp.rename(columns={f'Adj Close': 'adjusted_close'}, inplace=True)
      temp.columns = map(str.lower, temp.columns)
  else:
      temp.index = temp['timestamp']
      temp.index = pd.to_datetime(temp.index, infer_datetime_format=True)
      temp.drop('timestamp', axis = 1, inplace = True)

  temp = temp.loc[:, ['adjusted_close', 'high', 'close', 'open', 'low', 'volume']]
  return temp

def add_technical_features(data, window, fillna = True):
  inds = indicators_dict(data, window = window)
  df = pd.DataFrame(index = data.index)
  for key in inds.keys(): df[key] = inds[key]

  high, low, close, volume = data.high, data.low, data.close, data.volume
  df = df.join(add_momentum_indicators(high, low, close, volume, window, fillna))
  df = df.join(add_trend_indicators(high, low, close, volume, window, fillna))
  df = df.join(add_volatility_indicators(high, low, close, window, fillna))
  df = df.join(add_volume_indicators(high, low, close, volume, window, fillna))
  return df


