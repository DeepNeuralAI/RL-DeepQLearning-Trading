import streamlit as st
import altair as alt
import pandas as pd

from src.methods import evaluate_model
from src.agent import RLAgent
from src.utils import get_stock_data
from main.utils.plotting import visualize

import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

@st.cache
def load_data(symbol):
  temp = pd.read_csv(f'data/{symbol}.csv')

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

def load_model(state_size, model_name):
  return RLAgent(state_size = window_size, pretrained = True, model_name = model_name)

def evaluate(agent, test_data, window_size, verbose = True):
  result, history, shares = evaluate_model(agent, test_data, window_size, verbose)
  return result, history, shares

st.title('Run Model')
st.markdown('Subheading Here')

symbols = ['AAPL', 'AMZN', 'CRM', 'FB', 'GOOG', 'JNJ', 'JPM', 'MSFT', 'NFLX', 'SPY', 'V']
symbol = st.sidebar.selectbox('Stock Symbol:', symbols)

data = load_data(symbol)
start_date = st.sidebar.date_input('Start', data.index[-1])
end_date = st.sidebar.date_input('End', data.index[0])
window_size = st.sidebar.slider('Window Size', 1, 30, 10)
submit = st.sidebar.button('Run')

@st.cache
def filter_data_by_date(data, start_date, end_date):
  date_range = pd.date_range(start = start_date, end = end_date)
  return data.loc[date_range].dropna()

if submit:
  filtered_data = filter_data_by_date(data, start_date, end_date)

  st.write(filtered_data)
  trades = st.checkbox('Show Trades')




# model_name = 'model_double-dqn_GOOG_50'
#
#
# verbose = True

# test_stock = 'data/GOOG_2019.csv'
# test_data = get_stock_data(test_stock)
# window_size = 10
# agent = load_model(state_size = window_size, model_name = 'model_double-dqn_GOOG_50' )
# result, history, shares = evaluate(agent, test_data, window_size = window_size)
# chart = visualize(df, history)
# st.altair_chart(chart)

#
