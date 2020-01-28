import pandas as pd
import numpy as np
import ta


def add_technical_features(data):
  data = ta.utils.dropna(data)
  return ta.add_all_ta_features(
    data, open="open", high="high", low="low", close="adjusted_close",
    volume="volume", fillna=True)

def add_momentum_indicators(high, low, close, volume, n_period = 7, fillna=True):
	"""
	Money Flow Index (MFI)
	True strength index (TSI)
	Kaufman's Adaptive Moving Average (KAMA)
	"""
	df = pd.DataFrame(index = close.index)
	df['mom_mfi'] = ta.momentum.MFIIndicator(high, low, close, volume, n_period, fillna = fillna).money_flow_index()
	df['mom_tsi'] = ta.momentum.TSIIndicator(close, fillna = fillna).tsi()
	df['mom_kama'] = ta.momentum.kama(close, fillna=fillna)
	return df

def add_trend_indicators(high, low, close, volume, n_period = 7, fillna = True):
	"""
	Vortex Indicator (VI)
	Trix (TRIX)
	Mass Index (MI)
	Commodity Channel Index (CCI)
	Detrended Price Oscillator (DPO)
	KST Oscillator (KST)
	Ichimoku Kinkō Hyō (Ichimoku)
	Parabolic Stop And Reverse (Parabolic SAR)
	"""
	df = pd.DataFrame(index = close.index)
	df['trend_vi'] = ta.trend.VortexIndicator(high, low, close, n_period, fillna).vortex_indicator_diff()
	df['trend_trix'] = ta.trend.trix(close, n_period, fillna)
	df['trend_mi'] = ta.trend.MassIndex(high, low, fillna=fillna)
	df['trend_cci'] = ta.trend.CCIIndicator(high, low, close, fillna=fillna).cci()
	df['trend_dpo'] = ta.trend.DPOIndicator(close, n_period, fillna).dpo()
	df['trend_kst'] = ta.trend.kst_sig(close, fillna=fillna)
	df['trend_ichimoku'] = ta.trend.IchimokuIndicator(high, low, fillna=fillna).ichimoku_a()
	df['trend_sar'] = ta.trend.PSARIndicator(high, low, close, fillna=fillna).psar()
	return df

def add_volatility_indicators(high, low, close, n_period = 7, fillna=True):
	"""
	Average True Range (ATR)
	Keltner Channel (KC)
	Donchian Channel (DC)
	"""
	df = pd.DataFrame(index = close.index)
	df['vol_atr'] = ta.volatility.AverageTrueRange(high, low, close, n_period, fillna).average_true_range()
	df['vol_kc'] = ta.volatility.KeltnerChannel(high, low, close, n_period, fillna).keltner_channel_central()
	df['vol_dc_hband'] = ta.volatility.DonchianChannel(close, n_period, fillna = fillna).donchian_channel_hband()
	df['vol_dc_lband'] = ta.volatility.DonchianChannel(close, n_period, fillna = fillna).donchian_channel_lband()
	return df

def add_volume_indicators(high, low, close, volume, n_period = 7, fillna=True):
	"""
	Accumulation/Distribution Index (ADI)
	Chaikin Money Flow (CMF)
	Volume-price Trend (VPT)
	Negative Volume Index (NVI)
	"""
	df = pd.DataFrame(index = close.index)
	df['volume_adi'] = ta.volume.AccDistIndexIndicator(high, low, close, volume, fillna).acc_dist_index()
	df['volume_cmf'] = ta.volume.ChaikinMoneyFlowIndicator(high, low, close, volume, n_period, fillna).chaikin_money_flow()
	df['volume_vpt'] = ta.volume.VolumePriceTrendIndicator(close, volume, fillna).volume_price_trend()
	df['volume_nvi'] = ta.volume.NegativeVolumeIndexIndicator(close, volume, fillna).negative_volume_index()
	return df

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
