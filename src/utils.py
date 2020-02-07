import datetime as dt
import math
import logging
import pandas as pd
import numpy as np
import pdb
import plotly.express as px
import plotly.graph_objects as go

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
      temp.index.name = 'Date'
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

def results_df(price, shares, starting_value = 10_000):
  results = pd.DataFrame(columns = ['Price', 'Shares', 'Cash', 'Net_Cash', 'Net_Holdings', 'Value', 'Port_Vals'])
  results.Price = price
  results.Shares = shares
  results.Cash = np.zeros(results.shape[0])
  results.Cash = results.Shares * -results.Price
  results.Cash[0] += starting_value
  results.Net_Cash = results.Cash.cumsum()

  results.Net_Holdings = results.Shares.cumsum()
  results.Value = results.Price * results.Net_Holdings
  results.Port_Vals = results.Net_Cash + results.Value
  return results

def get_portfolio_stats(port_val, daily_rf = 0, samples_per_year = 252):
  cum_return = (port_val[-1] / port_val[0]) - 1
  daily_returns = (port_val /port_val.shift(1)) - 1
  daily_returns.iloc[0] = 0
  daily_returns = daily_returns[1:]

  avg_daily_returns = daily_returns.mean()
  std_daily_returns = daily_returns.std()

  K = np.sqrt(samples_per_year)
  sharpe_ratio = K * (avg_daily_returns - daily_rf) / std_daily_returns
  return cum_return, avg_daily_returns, std_daily_returns, sharpe_ratio

def plot_trades(data, trades, symbol):
  buy_x = trades.index[trades > 0]
  buy_y = data.price[trades > 0]

  sell_x = trades.index[trades < 0]
  sell_y = data.price[trades < 0]

  fig = px.line(data, x=data.index, y='price')
  fig.add_trace(go.Scatter(
    x=buy_x,
    y=buy_y,
    mode="markers",
    opacity = 0.8,
    marker = dict(size = 5, symbol = 0, color = 'lime',
      line=dict(width=1,color='DarkSlateGrey')
    ),
    name="Buy",
  ))
  fig.add_trace(go.Scatter(
    x=sell_x,
    y=sell_y,
    mode="markers",
    marker = dict(size = 5, symbol = 2, color = 'red'),
    name="Sell",
  ))
  fig.update_layout(
    xaxis_title="<b>Date</b>",
    yaxis_title='<b>Price</b>',
    legend_title='<b> Action </b>',
    template='plotly_white'
  )
  return fig


