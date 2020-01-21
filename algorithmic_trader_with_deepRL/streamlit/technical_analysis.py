import streamlit as st
import pandas as pd
import numpy as np
import sys
sys.path.append('./')
import os
from technical_indicators import add_technical_features, get_indicators_by_type, find_indexes, normalize
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import datetime as dt

def get_path(symbol, base_dir=None):
  if base_dir is None:
    base_dir = os.environ.get('PWD')
  return os.path.join(base_dir, f'../data/daily_adjusted_{symbol}.csv')

@st.cache
def get_df(symbol):
  cols = ['timestamp', 'high', 'low', 'adjusted_close', 'volume']
  data = pd.read_csv(get_path(symbol), index_col = 'timestamp', parse_dates=True, usecols = cols, na_values=['nan'])
  df = add_technical_features(data)
  return df

# Streamlit
symbols = ['AAPL', 'AMZN', 'CRM', 'FB', 'GOOG', 'JNJ', 'JPM', 'MSFT', 'NFLX', 'SPY', 'V']
symbol = st.sidebar.selectbox('Stock Symbol:', symbols)
data = get_df(symbol)

st.title('Technical Analysis')
start_date = st.date_input('Start', data.index[-1])
end_date = st.date_input('End', data.index[0])
date_range = pd.date_range(start_date, end_date)
st.write(type(date_range))

indicator_types = ['Trend', 'Momentum', 'Volatility', 'Volume', 'Others']
indicator_type = st.sidebar.selectbox('Indicator type:', indicator_types)
indicator_type_data = get_indicators_by_type(indicator_type.lower(), data)
indicators = indicator_type_data.columns.tolist()
selected_indicators = st.sidebar.multiselect("Indicator", indicators)
submit = st.sidebar.button('Plot')
raw = st.checkbox('Show DataFrame')

selected = data.loc[date_range].dropna(subset=['adjusted_close'])
source = selected.loc[:, selected_indicators]

if raw: st.dataframe(source)
if submit:
  pass

