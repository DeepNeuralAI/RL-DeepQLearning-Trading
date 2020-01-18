from technical_indicators import *
from get_data import load_data, get_path

def create_dataset(symbol, prices, volume):
  return {
    f'{symbol}': prices,
    'rsi': relative_strength_index(prices),
    'moms': momentum(prices),
    'stok': stochastic_oscillator_k(prices),
    'stod': stochastic_oscillator_d(prices),
    'obv': on_balance_volume(prices, volume),
    'sma': simple_moving_average(prices),
    'p2sma': price_to_sma(prices, simple_moving_average(prices)),
    'ema': exponential_moving_average(prices),
    'p2ema': price_to_ema(prices, exponential_moving_average(prices)),
    'macd': moving_average_convergence_divergence(prices),
    'bbl': bollinger_bands(prices, simple_moving_average(prices))[0],
    'bbh': bollinger_bands(prices, simple_moving_average(prices))[1],
    'bbp': bollinger_band_pct(prices, simple_moving_average(prices))
  }

def create_df(data, index):
  df = pd.DataFrame(index = index)
  for key in data.keys(): df[key] = data[key]
  return df

def get_df(symbol):
  cols = ['timestamp', 'high', 'low', 'adjusted_close', 'volume']
  data = pd.read_csv(get_path(symbol), index_col = 'timestamp', parse_dates=True, usecols = cols, na_values=['nan'])
  df = add_technical_features(data)
  return df



