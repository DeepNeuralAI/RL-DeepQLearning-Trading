import argparse
from src.utils import get_stock_data
from src.agent import RLAgent
from src.methods import evaluate_model
from src.utils import show_evaluation_result
import os
import coloredlogs
import keras.backend as K
import logging


def run(eval_stock, window_size, model_name, verbose):
  data = get_stock_data(eval_stock)
  initial_offset = data[1] - data[0]

  if model_name is not None:
    agent = RLAgent(window_size, pretrained=True, model_name=model_name)
    profit, history, shares = evaluate_model(agent, data, window_size, verbose)
    show_evaluation_result(model_name, profit, initial_offset)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Evaluate RLAgent')
  parser.add_argument('--eval')
  parser.add_argument('--window-size', default = 10)
  parser.add_argument('--model-name')
  parser.add_argument('--verbose', default = False)

  args = parser.parse_args()

  eval_stock = args.eval
  window_size = int(args.window_size)
  model_name = args.model_name
  verbose = args.verbose

  coloredlogs.install(level="DEBUG")

  if K.backend() == "tensorflow":
    logging.debug("Switching --> TensorFlow for CPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

  try:
    run(eval_stock, window_size, model_name, verbose)
  except KeyboardInterrupt:
    print("Aborted with Keyboard Interrupt..")
