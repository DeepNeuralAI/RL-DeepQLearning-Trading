# DeepRL Trader

![demo](public/demo.gif)

This project frames stock market trading as a _Markov Decision Process._ Specifically, this application uses the deep reinforcement learning model [**Double Deep Q Network**](https://arxiv.org/abs/1509.06461) to generate an optimal set of trades that maximizes daily return.

## Main Idea
 This application takes a model free approach and develop a variation of Deep Q-Learning to estimate the optimal actions of a trader.

 The model is a FCN trained using *experience replay* and *Double DQN* with input features given by the current state of the limit order book, 33 additional technical indicators, and available execution actions, while the output is the Q-value function estimating the future rewards under an arbitrary action.

 We apply the model to ten stocks and observe that it does on occasion outperform the standard benchmark approach on most stocks using the measure of Sharpe Ratio.

#### Presentation Slides

Further details regarding the motivation, methods and results of implementation can be found in my presentation [here](http://bit.ly/Aaron-Mendonsa-DeepRLSlides).

## Usage

1. To play interactively with the model, visit the deployed Streamlit app [here](http://bit.ly/DeepRLTrader)
2. To run it locally:
```shell
git clone https://github.com/DeepNeuralAI/RL-DeepQLearning-Trading.git
pip install -r requirements.txt
streamlit run app.py
```

### Training

To train the model, use the following command:
```shell
$ python3 train.py --train data/GOOG.csv --valid GOOG_2018.csv --episode-count 50 --window-size 10
```

### Evaluation

To evaluate the given model, use the following command:
```shell
$ python3 evaluate.py --eval data/GOOG.csv --model-name GOOG --window-size 10 --verbose True:
```

## How It Works


