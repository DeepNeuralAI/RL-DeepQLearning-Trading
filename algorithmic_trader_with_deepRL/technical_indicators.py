import pandas as pd
import numpy as np
import ta


def add_technical_features(data):
  data = ta.utils.dropna(data)
  return ta.add_all_ta_features(
    data, open="open", high="high", low="low", close="adjusted_close",
    volume="volume", fillna=True)

def find_indexes(indicator_type, columns):
  indicator = f'{indicator_type}_'
  indexes = []
  for index, col in enumerate(columns):
      if indicator in col: indexes.append(index)
  return indexes

def get_indicators_by_type(indicator_type, data):
  cols = data.columns.values
  indexes = find_indexes(indicator_type, cols)
  indicators = cols[indexes]
  return data.loc[:, indicators]

def normalize(prices):
	return prices / prices.values[0]

# momentum
def relative_strength_index(prices, window = 7):
	diff = prices.diff().fillna(0)
	gain, loss = diff.copy(), diff.copy()

	gain[gain < 0] = 0
	loss[loss > 0] = 0

	rs_up = exponential_moving_average(gain, window)
	rs_down = exponential_moving_average(loss.abs(), window)
	rs = rs_up / rs_down
	rsi = 100 - (100 / (1 + rs))
	return rsi.fillna(method='bfill')

def momentum(prices, window = 7):
	moment = (prices / prices.shift(window)) - 1
	return moment.fillna(method='bfill')

def stochastic_oscillator_k(prices, window = 7):
	rolling_max = prices.rolling(window).max().fillna(method='bfill')
	rolling_min = prices.rolling(window).min().fillna(method='bfill')
	return (prices - rolling_min) / (rolling_max - rolling_min)


def stochastic_oscillator_d(prices, k_window = 7, d_window = 3):
	stok = stochastic_oscillator_k(prices, k_window)
	return stok.rolling(window = d_window).mean().fillna(method ='bfill')

# volume
def on_balance_volume(prices, volume):
	vol = prices.diff().fillna(0).apply(np.sign) * volume
	vol.iloc[0] = 1.
	return vol.cumsum()

# trend
def simple_moving_average(prices, window = 7):
	sma = prices.rolling(window).mean()
	sma = sma.fillna(method='bfill')
	return sma

def price_to_sma(prices, sma):
	return prices / sma

def exponential_moving_average(prices, window = 7):
	return prices.ewm(span=window, adjust=False).mean()

def price_to_ema(prices, ema):
	return prices / ema

def moving_average_convergence_divergence(prices, short_window = 7, long_window = 14):
	short = prices.ewm(span = short_window, adjust = False).mean()
	long = prices.ewm(span = long_window, adjust = False).mean()
	return short - long

# volatility
def bollinger_bands(prices, sma):
	std = prices.std(axis = 0)
	upper = sma + 2*std
	lower = sma - 2*std
	return lower, upper

def bollinger_band_pct(prices, sma):
	lower, upper = bollinger_bands(prices, sma)
	return (prices - lower) / (upper - lower)
