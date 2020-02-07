import streamlit as st
import pandas as pd

from src.utils import (
  load_data,
  add_technical_features,
  show_evaluation_result,
  normalize,
  results_df,
  get_portfolio_stats,
  plot_trades,
  plot_benchmark
)
from PIL import Image

def how_it_works():

  st.header('How It Works')

  st.subheader('Reinforcement Learning Primer')

  st.write('We will frame market trading in a reinforcement learning context.')
  rl_diagram = Image.open('public/rl_diagram.png')
  st.image(rl_diagram, caption='Reinforcement Learning Process', use_column_width=True)

  st.markdown('1. The Agent observes the environment, in the form of a state \n 2. Based on that state, the Agent takes a certain action based upon a policy \n 3. For that given action, and state, the Agent receives a reward from the environment. \n 4. The action mutates the environment to transition to a new state. \n 5. Repeat.')

  st.markdown('Q-learning is a model-free algorithm in RL for the purpose of learning a policy. The policy of an agent is arguably the most important as it is the policy that drives how the agent interacts with its environment. We define the "goodness" of an action by using the mathematical action-value function **Q(s,a)**. The higher the Q-value, the higher probability that given action _a_ in state _s_ will bring a higher reward _r_.')

  st.markdown('We can use a table to store experience tuples, namely a _Q-table_, to take a discrete input of state _s_ and action _a_  and output an associated Q-value. The one limitation of this method, despite its intuitiveness, is the scalability. With continuous states such as a stock price, the computational space would be inefficient to store _n_ states by _m_ actions. Chess for example would take a 10^120 size states space.')

  st.write('Instead of storing a massive lookup table, we can instead approximate Q(s,a) with neural networks, named a Deep Q Network (DQN)')


  dqn = Image.open('public/dqn.png')
  st.image(dqn, caption = 'Using a Deep Q Network can approximate Q(s,a)', use_column_width = True)

  st.write('In 2015, Google DeepMind showed that in stochastic environments, Q-learning and DQN tends to overestimate and learn very poorly. From a high level perspective, these overestimations tend to result from a positive bias due to taking the maximum expected action value. Hasselt, et.al proposed using a double estimator to construct DQN and showed that the Double DQN (DDQN) converged to a more optimal policy and tended to estimate the true value more closely.')

  estimate = Image.open('public/ddqn_estimate.png')
  st.image(estimate, use_column_width = True, caption = 'DQN tends to overestimate action values')

  st.subheader('Data Process')
  st.write('Time series daily data is extracted via API request from Alpha Vantage. Example Google financial data extracted for a given time period shown below:')

  @st.cache
  def load_data_(symbol, window_size):
    data_ = add_technical_features(load_data(f'data/{symbol}.csv'), window = window_size).sort_values(by=['Date'], ascending=True)
    return data_

  data = pd.read_csv('data/GOOG.csv')
  st.dataframe(data.head())
  st.markdown('From the above data example, feature generation occurs.\n Technical indicators are derived from fundamental price and volume in the categories of:')
  st.markdown('* Trend \n * Momentum \n* Volatility \n* Volume')
  st.write('The final dataframe with a total of 33 included technical indicators is shown below:')
  st.dataframe(load_data_('GOOG', 10).head())
  st.markdown('The above example is then normalized and fed through the Double Deep Q network that will be discussed below. ')
  st.markdown('#### Training Data')
  st.write('The RL agent is trained on 7-10 years of historical data.')
  st.markdown('#### Test Data')
  st.write('The RL agent is tested on an unseen set of 1-2 years of price/volume data. In most examples, this would be 2019 price/volume data')

  st.subheader('Model')


  st.subheader('Results')
