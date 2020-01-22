import sys
import pandas as pd
import numpy as np
from util import data_df

class BaselineModel():
  def __init__(self, symbol, start_date, end_date, max_shares = 100):
    self.symbol = symbol
    self.start_date, self.end_date = start_date, end_date
    self.max_shares = max_shares
    self.data = self.get_data()
    self.shares = self.shares_df()
    self.trades = self.trades_df()

  def get_data(self):
    data = data_df(self.symbol).loc[:, self.symbol]
    date_range = pd.date_range(start = self.start_date, end = self.end_date)
    return data.loc[date_range].dropna()

  def shares_df(self):
    shares = np.zeros(self.data.shape[0])
    shares[0] = self.max_shares
    return shares[:, None]

  def trades_df(self):
    trades = pd.DataFrame(0.0, columns=[self.symbol, 'Cash'], index=self.data.index)
    current_value = self.shares * (-self.data[:, None])
    trades[self.symbol] = self.shares
    trades['Cash'] += current_value.ravel()
    return trades
