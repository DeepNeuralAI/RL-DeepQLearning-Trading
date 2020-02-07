import pandas as pd
import numpy as np

class HeuristicTrader():
  def __init__(self, symbol, data, window = 7, max_shares = 10):
    self.symbol = symbol
    self.window, self.max_shares = window, max_shares
    self.data = data
    self.policy = self.strategy()
    self.shares = self.shares_df()


  def strategy(self):
    tmp_df = pd.DataFrame(index = self.data.index)
    tmp_df['indicator'] = np.zeros(self.data.shape[0])

    buy = (self.data['vol_bbp'] < 0.5) & (self.data['trend_rsi'] <= 30) & \
        (self.data['trend_p2sma'] <= self.data['trend_p2sma'].quantile(0.25))
    sell = (self.data['vol_bbp'] >= 0.5) & (self.data['trend_rsi'] > 50) & \
        (self.data['trend_p2sma'] >= self.data['trend_p2sma'].quantile(.75))

    tmp_df['indicator'][buy] = 1
    tmp_df['indicator'][sell] = -1
    return tmp_df

  def shares_df(self):
    shares = []
    net_holdings = 0

    for _, row in self.policy.iterrows():
      buy = row['indicator'] > 0
      sell = row['indicator'] < 0

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
