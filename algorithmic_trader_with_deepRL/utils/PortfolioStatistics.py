import pandas as pd
import numpy as  np

class PortfolioStatistics():
  def __init__(self, prices, trades, starting_balance = 1_000, daily_rf = 0, samples_per_year = 252):
    self.trades = trades
    self.prices = prices
    self.starting_balance = starting_balance
    self.daily_rf, self.samples_per_year = daily_rf, samples_per_year
    self.holdings = self.holdings_df()
    self.stock_values = self.stock_value_df()
    self.cr, self.adr, self.sdr, self.sr = self.portfolio_statistics()

  def holdings_df(self):
    holdings = self.trades.copy()
    holdings['Cash'][0] += self.starting_balance
    return holdings.cumsum()

  def stock_value_df(self):
    holdingsAdj = self.holdings.drop('Cash', axis = 1)
    vals = self.prices[:, None] * holdingsAdj
    vals['Cash'] = self.holdings[['Cash']]
    return vals.sum(axis = 1)

  def portfolio_statistics(self):
    cum_return = (self.stock_values[-1] / self.stock_values[0]) - 1
    daily_returns = (self.stock_values/ self.stock_values.shift(1)) - 1
    daily_returns.iloc[0] = 0
    daily_returns = daily_returns[1:]

    avg_daily_returns = daily_returns.mean()
    std_daily_returns = daily_returns.std()

    K = np.sqrt(self.samples_per_year)
    sharpe_ratio = K * (avg_daily_returns - self.daily_rf) / std_daily_returns
    return cum_return, avg_daily_returns, std_daily_returns, sharpe_ratio
