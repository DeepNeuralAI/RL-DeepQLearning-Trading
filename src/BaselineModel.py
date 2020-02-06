import pandas as pd
import numpy as np
import pdb

class BaselineModel():
  def __init__(self, symbol, data, max_shares = 10):
    self.symbol = symbol
    self.data = data
    self.max_shares = max_shares
    self.shares = self.shares_df()
    self.trades = self.trades_df()

  def shares_df(self):
    shares = np.zeros(self.data.shape[0])
    shares[0] = self.max_shares
    shares[-1] = -self.max_shares
    return shares[:, None]

  def trades_df(self):
    trades = pd.DataFrame(0.0, columns=[self.symbol, 'Cash'], index=self.data.index)
    current_value = self.shares * (-self.data.price[:, None])
    trades[self.symbol] = self.shares
    trades['Cash'] += current_value.ravel()
    return trades
