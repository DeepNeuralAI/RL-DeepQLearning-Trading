from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
import numpy as np
import pandas as pd

class RNN:
  def __init__(self, data, lag = 10, horizon = 10, num_features = 1, epochs = 50):
    self.data = data
    self.lag = lag
    self.horizon = horizon
    self.num_features = num_features
    self.X_train, self.y_train = self.load_data()
    self.model = self.model_()
    self.epochs = epochs

  def load_data(self):
    df = pd.DataFrame(self.data)
    data = df.as_matrix()
    lags = []
    horizons = []
    nsample = len(data) - self.lag - self.horizon

    for i in range(nsample):
      lags.append(data[i: i + self.lag , -self.num_features:])
      horizons.append(data[i + self.lag : i + self.lag + self.horizon, -1])

    lags = np.array(lags)
    horizons = np.array(horizons)
    lags = np.reshape(lags, (lags.shape[0], lags.shape[1], self.num_features))
    return lags, horizons

  def model_(self):
    model = Sequential()
    model.add(LSTM(units=50,return_sequences=True,input_shape=(self.X_train.shape[1], self.num_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=self.X_train.shape[1]))
    model.compile(optimizer='adam',loss='mean_squared_error')
    return model

  def train(self):
    self.model.fit(self.X_train, self.y_train, epochs = self.epochs, batch_size = 32)




