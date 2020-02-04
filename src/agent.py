import random
from collections import deque
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential
from keras.models import load_model, clone_model
from keras.layers import Dense
from keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from src.utils import timestamp
import pdb

class RLAgent:
  def __init__(self, state_size, model_type = 'ddqn', pretrained = False, model_name = None, window_size = 10, reset_target_weight_interval = 10):
    self.model_type = model_type

    self.state_size = state_size
    self.action_size = 3
    self.inventory = []
    self.memory = deque(maxlen = 10000)
    self.start = True

    self.model_name = model_name
    self.gamma = 0.99
    self.rar = 0.99 # Epsilon / Random Action Rate
    self.eps_min = 0.01
    self.radr = 0.995 # Random Action Decay Rate
    self.lr = 1e-5
    self.loss = Huber
    self.custom_objects = {"huber": Huber}
    self.optimizer = Adam(lr = self.lr)
    self.window_size = window_size

    if pretrained and self.model_name is not None:
      self.model = self.load()
    else:
      self.model = self.model_()

    self.n_iter = 1
    self.reset_interval = reset_target_weight_interval

    self.target_model = clone_model(self.model)
    self.target_model.set_weights(self.model.get_weights())


  def load(self):
    model = load_model(f"models/{self.model_name}", custom_objects = self.custom_objects, compile=False)
    model.compile(optimizer = self.optimizer, loss = self.loss())
    return model

  def save(self, episode):
    if self.model_name is None:
      self.model_name = f'{self.model_type}_{timestamp()}'
    self.model.save(f"models/{self.model_name}_{episode}")

  def model_(self):
    model = Sequential()
    model.add(Dense(units=256, activation="relu", input_shape=(self.state_size,)))
    model.add(Dense(units=512, activation="relu"))
    model.add(Dense(units=512, activation="relu"))
    model.add(Dense(units=256, activation="relu"))
    model.add(Dense(units=self.action_size))

    model.compile(optimizer = self.optimizer, loss = self.loss())
    return model

  def action(self, state, evaluation = False):
    if self.start:
      self.start = False
      return 1

    if not evaluation and (random.random() <= self.rar):
      return random.randrange(self.action_size)

    action_probs = self.model.predict(state)

    if evaluation:
      print(action_probs[0])
    return np.argmax(action_probs[0])

  def replay(self, batch_size):
    mini_batch = random.sample(self.memory, batch_size)
    X_train, y_train = [], []

    if self.model_type == 'ddqn':
      if self.n_iter % self.reset_interval == 0:
        self.target_model.set_weights(self.model.get_weights())

      for state, action, reward, next_state, done in mini_batch:
        if done:
          target = reward
        else:
          target = reward + self.gamma * self.target_model.predict(next_state)[0][np.argmax(self.model.predict(next_state)[0])]

        q_values = self.model.predict(state)
        q_values[0][action] = target
        X_train.append(state[0])
        y_train.append(q_values[0])

    if self.rar > self.eps_min:
      self.rar *= self.radr

    loss = self.model.fit(
      x  = np.array(X_train),
      y = np.array(y_train),
      epochs = 1,
      verbose = 0
    ).history["loss"][0]

    return loss

  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))












