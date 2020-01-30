import argparse
import logging
import coloredlogs
from docopt import docopt
import datetime as dt
import keras.backend as K
import numpy as np

from src.utils import timestamp, show_training_result, get_stock_data
from src.methods import train_model, evaluate_model
from src.agent import RLAgent
from src.rnn import RNN
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pdb


def run(training_stock, validation_stock, window_size, batch_size, episode_count, model_type="ddqn", model_name = None, pretrained = False, verbose = False, mode = None):
  training_data = get_stock_data(training_stock)
  validation_data = get_stock_data(validation_stock)

  # if mode == 'rnn':
  #   if model_name is None:
  #     data_ = np.array(training_data) / np.array(training_data)[0]
  #     rnn = RNN(data_, lag = window_size, horizon = window_size, epochs = 100)
  #     rnn.train()
  #     rnn.model.save('models/rnn_100')
  #     print('Saved..')
  #     return

  agent = RLAgent(window_size, model_type = model_type, model_name = model_name)

  initial_offset = validation_data[1] - validation_data[0]

  for episode in range(1, episode_count + 1):
    training_result = train_model(agent, episode, training_data, episode_count = episode_count, batch_size = batch_size, window_size = window_size, mode = mode)
    validation_result, _, shares = evaluate_model(agent, validation_data, window_size, verbose, mode = mode)
    show_training_result(training_result, validation_result, initial_offset)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Deep RL in Algo Trading')
  parser.add_argument('--train')
  parser.add_argument('--valid')
  parser.add_argument('--model-type', default = 'ddqn')
  parser.add_argument('--window-size', default = 10)
  parser.add_argument('--batch-size', default = 32)
  parser.add_argument('--episode-count', default = 50)
  parser.add_argument('--model-name')
  parser.add_argument('--pretrained', default = False)
  parser.add_argument('--verbose', default = False)
  parser.add_argument('--mode')

  args = parser.parse_args()

  training_stock = args.train
  validation_stock = args.valid
  model_type = args.model_type
  window_size = int(args.window_size)
  batch_size = int(args.batch_size)
  episode_count = int(args.episode_count)
  mode = args.mode

  model_name = args.model_name
  pretrained = args.pretrained
  verbose = args.verbose

  coloredlogs.install(level = "DEBUG")

  if K.backend() == "tensorflow":
    logging.debug("Switching --> TensorFlow for CPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

  try:
    run(training_stock, validation_stock, window_size, batch_size, episode_count,
    model_type=model_type, pretrained = pretrained, verbose = verbose, mode = mode)
  except KeyboardInterrupt:
    print("Aborted with Keyboard Interrupt..")
