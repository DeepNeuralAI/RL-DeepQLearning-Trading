import sys
import datetime as dt
import pandas as pd
import numpy as np
from util import data_df

class PerfectTrader():
  def __init__(self, symbol, start_date, end_date, max_shares = 100):
    self.symbol = symbol
    self.start_date, self.end_date = start_date, end_date
    self.max_shares = max_shares
    self.data = self.get_data()
    self.policy = self.strategy()
    self.shares = self.shares_df()
    self.trades = self.trades_df()

  def get_data(self):
    data = data_df(self.symbol).loc[:, self.symbol]
    date_range = pd.date_range(start = self.start_date, end = self.end_date)
    return data.loc[date_range].dropna()

  def strategy(self):
    return np.sign((self.data.shift(-1) - self.data).fillna(0))

  def shares_df(self):
    shares = []
    net_holdings = 0

    for row in self.policy:
      buy = row > 0
      sell = row < 0

      if buy and net_holdings == 0:
        shares.append(self.max_shares)
        net_holdings += self.max_shares
      elif buy and net_holdings == -self.max_shares:
        shares.append(self.max_shares * 2)
        net_holdings += self.max_shares * 2
      elif sell and net_holdings == 0:
        shares.append(-self.max_shares)
        net_holdings += -self.max_shares
      elif sell and net_holdings == self.max_shares:
        shares.append(-self.max_shares * 2)
        net_holdings += -self.max_shares * 2
      else:
        shares.append(0)
    return shares

  def trades_df(self):
    trades = pd.DataFrame(0.0, columns=[self.symbol, 'Cash'], index=self.data.index)
    current_value = self.shares * (-self.data)
    trades[self.symbol] = self.shares
    trades['Cash'] += current_value.ravel()
    return trades
