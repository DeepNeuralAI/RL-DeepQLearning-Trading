import streamlit as st
import pandas as pd
import numpy as np
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

from src.methods import evaluate_model
from src.agent import RLAgent
from src.BaselineModel import BaselineModel
from src.HeuristicTrader import HeuristicTrader


from src.utils import (
  load_data,
  add_technical_features,
  show_evaluation_result,
  normalize,
  results_df,
  get_portfolio_stats,
  plot_trades
)

@st.cache
def load_data_(symbol, window_size):
  data_ = add_technical_features(load_data(f'data/{symbol}.csv'), window = window_size).sort_values(by=['Date'], ascending=True)
  return data_

@st.cache
def filter_data_by_date(data, start_date, end_date):
  date_range = pd.date_range(start = start_date, end = end_date)
  return data.loc[date_range].dropna()

def load_model(state_size, model_name):
  return RLAgent(state_size = window_size, pretrained = True, model_name = model_name)

def evaluate(agent, test_data, window_size, verbose = True):
  result, history, shares = evaluate_model(agent, test_data, window_size, verbose)
  return result, history, shares

def sidebar(index):
  start_date = st.sidebar.date_input('Start', index[0])
  end_date = st.sidebar.date_input('End', index[-1])
  window_size = st.sidebar.slider('Window Size', 1, 30, 10)
  return start_date, end_date, window_size

def benchmarks(symbol, data, shares):
  pass

# Streamlit App
st.title('DeepRL Trader')
st.subheader('Model uses Double Deep Q Network to generate a policy of optimal trades')

symbols = ['AAPL', 'AMZN', 'CRM', 'FB', 'GOOG', 'JNJ', 'JPM', 'MSFT', 'NFLX', 'SPY', 'TSLA', 'V']
symbol = st.sidebar.selectbox('Stock Symbol:', symbols)

index = load_data_(symbol, 10).index
start_date, end_date, window_size = sidebar(index)
submit = st.sidebar.button('Run')


if submit:
  model_name = symbol
  data = load_data_(symbol, window_size)
  filtered_data = filter_data_by_date(data, start_date, end_date)


  agent = load_model(filtered_data.shape[1], model_name = model_name)
  profit, history, shares = evaluate(agent, filtered_data, window_size = window_size, verbose = False)
  results = results_df(filtered_data.price, shares, starting_value = 1_000)
  cum_return, avg_daily_returns, std_daily_returns, sharpe_ratio = get_portfolio_stats(results.Port_Vals)

  st.write(f'### Cumulative Return for {symbol}: {np.around(cum_return * 100, 2)}%')
  fig = plot_trades(filtered_data, results.Shares, symbol)
  st.plotly_chart(fig)


  ## Benchmarking
  baseline = BaselineModel(symbol, filtered_data, max_shares = 10)
  baseline_results = results_df(filtered_data.price, baseline.shares, starting_value = 1_000)

  heuristic = HeuristicTrader(symbol, filtered_data, window = window_size, max_shares = 10)
  heuristic_results = results_df(filtered_data.price, heuristic.shares, starting_value = 1_000)

  cum_return_base, avg_daily_returns_base, std_daily_returns_base, sharpe_ratio_base = get_portfolio_stats(baseline_results.Port_Vals)
  cum_return_heuristic, avg_daily_returns_heuristic, std_daily_returns_heuristic, sharpe_ratio_heuristic = get_portfolio_stats(heuristic_results.Port_Vals)

  benchmark = pd.DataFrame(columns = ['Cumulative Return', 'Avg Daily Returns', 'Std Dev Daily Returns', 'Sharpe Ratio'], index = ['Double DQN', 'Buy & Hold', 'Heuristic'])
  benchmark.loc['Double DQN'] = [cum_return * 100, avg_daily_returns * 100, std_daily_returns, sharpe_ratio]
  benchmark.loc['Heuristic' ] = [cum_return_heuristic * 100, avg_daily_returns_heuristic * 100, std_daily_returns_heuristic, sharpe_ratio_heuristic]
  benchmark.loc['Buy & Hold'] = [cum_return_base * 100, avg_daily_returns_base * 100, std_daily_returns_base, sharpe_ratio_base]


  st.table(benchmark.astype('float64').round(4))

  st.header('Raw Data')
  st.subheader('Double DQN')
  st.dataframe(results)

  st.subheader('Buy & Hold')
  st.write(baseline_results)

  st.subheader('Heuristic')
  st.write(heuristic_results)



