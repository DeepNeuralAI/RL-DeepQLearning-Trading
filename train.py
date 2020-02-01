import argparse
import logging
import coloredlogs
from docopt import docopt
import datetime as dt
import keras.backend as K
from src.utils import timestamp, show_training_result, load_data, add_technical_features
from src.methods import train_model, evaluate_model
from src.agent import RLAgent
from tensorboardX import SummaryWriter
import os
import pdb


def run(training_stock, validation_stock, window_size, batch_size, episode_count, model_type="ddqn", model_name = None, pretrained = False, verbose = False):
  writer = SummaryWriter()
  training_data = add_technical_features(load_data(training_stock), window = window_size)
  validation_data = add_technical_features(load_data(validation_stock), window = window_size)

  num_features = training_data.shape[1]
  agent = RLAgent(state_size = num_features, model_type = model_type, model_name = model_name, window_size = window_size)

  for episode in range(1, episode_count + 1):
    training_result = train_model(agent, episode, training_data, episode_count = episode_count, batch_size = batch_size, window_size = window_size)
    pdb.set_trace()
    writer.add_scalar('train/reward', training_result[2], episode)
    writer.add_scalar('train/loss', training_result[-1], episode)

    validation_result, _, shares = evaluate_model(agent, validation_data, verbose)
    writer.add_scaler('valid/reward', validation_result[2], episode)
    writer.add_scaler('valid/loss', validation_result[-1], episode)

    show_training_result(training_result, validation_result)



  writer.close()

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
