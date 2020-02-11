# DeepRL Trader

![trader](public/benchmark.png)

This project frames stock market trading as a _Markov Decision Process._ Specifically, this application uses a deep reinforcement learning model, **Double Deep Q Network** to generate an optimal set of trades that maximizes daily return.

## Instructions

1. To play interactively with the model, visit the deployed Streamlit app [here](http://bit.ly/DeepRLTrader)
2. To run it locally:
```python
git clone https://github.com/DeepNeuralAI/RL-DeepQLearning-Trading.git
pip install -r requirements.txt
streamlit run app.py
```

#### Presentation Slides

Further details regarding the motivation, methods and results of implementation can be found in my presentation [here](http://bit.ly/Aaron-Mendonsa-DeepRLSlides).

### Training

To train the model, use the following command:
```
$ python3 train.py --train data/GOOG.csv --valid GOOG_2018.csv --episode-count 50 --window-size 10
```

### Evaluation

To evaluate the given model, use the following command:
```
$ python3 evaluate.py --eval data/GOOG.csv --model-name GOOG --window-size 10 --verbose True:
```

## How It Works

### Reinforcement Learning Primer
<img src ="public/rl_diagram.png" height=300 width=500>

1. The Agent observes the environment, in the form of a state
2. Based on that state, the Agent takes a certain action based upon a policy
3. For that given action, and state, the Agent receives a reward from the environment.
4. The action mutates the environment to transition to a new state.
5. Repeat


### Q-Learning
Q-learning is a model-free algorithm in RL for the purpose of learning a policy. The policy of an agent is arguably the most important as it is the policy that drives how the agent interacts with its environment. We define the "goodness" of an action by using the mathematical action-value function **Q(s,a)**. The higher the Q-value, the higher probability that given action _a_ in state _s_ will bring a higher reward _r_.

We can use a table to store experience tuples, namely a _Q-table_, to take a discrete input of state _s_ and action _a_  and output an associated Q-value. The one limitation of this method, despite its intuitiveness, is the scalability. With continuous states such as a stock price, the computational space would be inefficient to store _n_ states by _m_ actions. Chess for example would take a 10^120 size states space.

Instead of storing a massive lookup table, we can instead approximate Q(s,a) with neural networks, named a Deep Q Network (DQN)

<img src="public/dqn.png" height=500 width=700>



### Data

### Model

## Results
