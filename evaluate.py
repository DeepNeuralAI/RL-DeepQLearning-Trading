import argparse
from src.utils import get_stock_data
from src.agent import RLAgent
from src.methods import evaluate_model
from src.utils import show_evaluation_result, load_data, add_technical_features
import os
import coloredlogs
import keras.backend as K
import logging
import pdb


def run(eval_stock, window_size, model_name, verbose):
  data = add_technical_features(load_data(eval_stock), window = window_size)
  num_features = data.shape[1]

  if model_name is not None:
    agent = RLAgent(num_features, pretrained=True, model_name=model_name)
    profit, history, shares, cum_return = evaluate_model(agent, data, verbose)
    show_evaluation_result(profit)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Evaluate RLAgent')
  parser.add_argument('--eval')
  parser.add_argument('--window-size', default = 10)
  parser.add_argument('--model-name')
  parser.add_argument('--verbose', default = True)


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
