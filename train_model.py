import logging
import coloredlogs
from docopt import docopt
import datetime as dt
import keras.backend as K
from src.utils import timestamp, show_training_result, get_stock_data
from src.methods import train_model, evaluate_model
from src.agent import RLAgent
import os


def run(training_stock, validation_stock, window_size, batch_size, episode_count, model_type="ddqn", model_name=f"model_{timestamp()}", pretrained = False, verbose = False):
  agent = RLAgent(window_size, model_type = model_type, model_name = model_name)

  training_data = get_stock_data(training_stock)
  validation_data = get_stock_data(validation_stock)

  initial_offset = validation_data[1] - validation_data[0]

  for episode in range(episode_count):
    training_result = train_model(agent, episode, training_data, episode_count = episode_count + 1, batch_size = batch_size, window_size = window_size)
    validation_result, _, shares = evaluate_model(agent, validation_data, window_size, verbose)
    show_training_result(training_result, validation_result, initial_offset)

if __name__ == "__main__":
  args = docopt(__doc__)

  training_stock = args["<training-stock>"]
  validation_stock = args["<validation-stock>"]
  model_type = args["--model-type"]
  window_size = args["--window-size"]
  batch_size = args["--batch-size"]
  episode_count = args["--episode-count"]

  model_name = args["--model-name"]
  pretrained = args["--pretrained"]
  verbose = args["--verbose"]

  coloredlogs.install(level = "DEBUG")

  if K.backend() == "tensorflow":
    logging.debug("Switching --> TensorFlow for CPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

  try:
    run(training_stock, validation_stock, window_size, batch_size, episode_count,
    model_type=model_type, model_name=model_name, pretrained = pretrained, verbose = verbose)
  except KeyboardInterrupt:
    print("Aborted with Keyboard Interrupt..")
