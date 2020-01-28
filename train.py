import argparse
import logging
import coloredlogs
from docopt import docopt
import datetime as dt
import keras.backend as K
from src.utils import timestamp, show_training_result, get_stock_data
from src.methods import train_model, evaluate_model
from src.agent import RLAgent
import os


def run(training_stock, validation_stock, window_size, batch_size, episode_count, model_type="ddqn", model_name = None, pretrained = False, verbose = False):
  agent = RLAgent(window_size, model_type = model_type, model_name = model_name)

  training_data = get_stock_data(training_stock)
  validation_data = get_stock_data(validation_stock)

  initial_offset = validation_data[1] - validation_data[0]

  for episode in range(1, episode_count + 1):
    training_result = train_model(agent, episode, training_data, episode_count = episode_count, batch_size = batch_size, window_size = window_size)
    validation_result, _, shares = evaluate_model(agent, validation_data, window_size, verbose)
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

  args = parser.parse_args()

  training_stock = args.train
  validation_stock = args.valid
  model_type = args.model_type
  window_size = int(args.window_size)
  batch_size = int(args.batch_size)
  episode_count = int(args.episode_count)

  model_name = args.model_name
  pretrained = args.pretrained
  verbose = args.verbose

  coloredlogs.install(level = "DEBUG")

  if K.backend() == "tensorflow":
    logging.debug("Switching --> TensorFlow for CPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

  try:
    run(training_stock, validation_stock, window_size, batch_size, episode_count,
    model_type=model_type, pretrained = pretrained, verbose = verbose)
  except KeyboardInterrupt:
    print("Aborted with Keyboard Interrupt..")
