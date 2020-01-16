# Momentum
def relative_strength_index(prices, window = 7):
  	diff = prices.diff().fillna(0)
    gain, loss = diff.copy(), diff.copy()

    gain[gain < 0] = 0
    loss[loss > 0] = 0

    RS_up = exponential_moving_average(gain, window)
    RS_down = exponential_moving_average(loss.abs(), window)
    RS = RS_up / RS_down
    RSI = 100 - (100 / (1 + RS))
    return RSI.fillna(method='bfill')

def momentum(prices, window = 7):
  moment =(prices / prices.shift(window)) - 1
  return moment.fillna(method='bfill')

def stochastic_oscillator():
  pass

# Volume
def on_balance_volume():
  pass

# Trend
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

def moving_average_convergence_divergence():
  pass

# Volatility
def bollinger_bands(prices, sma):
  std = prices.std(axis = 0)
  upper = sma + 2*std
  lower = sma - 2*std
  return lower, upper

def bollinger_band_pct(prices, sma):
  lower, upper = bollinger_bands(prices, sma)
  return (prices - lower) / (upper - lower)


