# AlphaZero Implementation and Framework

This project is meant to be a reimplementation of the [AlphaZero](https://arxiv.org/abs/1712.01815) deep reinforcement learning algortihm. In addition, the project aims to provide a framework where users can define their own models and play arbitrary games using the algorithm.

## Description

AlphaZero is a deep reinforcement learning algorithm developed by DeepMind, a subsidiary of Google. It is designed to learn how to play games such as chess, shogi, and Go at a superhuman level, without any prior knowledge of the game other than the rules.

The algorithm uses a combination of Monte Carlo tree search, neural network training, and self-play to improve its performance over time. Initially, it makes random moves and learns from its mistakes through trial and error. As it becomes more experienced, it uses a neural network to evaluate positions and determine the best move to make in a given situation.

AlphaZero has been able to defeat some of the world's strongest human players in games like chess and Go, and has been hailed as a breakthrough in artificial intelligence research. Its success has also led to new advances in machine learning and reinforcement learning algorithms.

This project aims to reimplement AlphaZero as closely to the original paper as possible (including things like asynchronous Monte Carlo tree search and parallelism), but with our limited resources we don't expect to get the same results that the original paper did.

## Roadmap

1. Perform one full iteration of the AlphaZero algorithm without parallelism.

## Getting Started

### Dependencies

Python
CUDA drivers (if using Nvidia GPU)

### Installing
We recommend running and installing dependencies for this project in a [virtual environment](https://docs.python.org/3/library/venv.html).

```
python -m pip install -r requirements.txt
```

### Executing program

1. Download dependencies
2. Navigate to the src/ folder
3. Run the program using the command below
```
python run.py
```

## Help

```
```

## Authors

[Nathaniel Nemenzo](nnemenzo.com)

## Version History

```
```

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments

* [AlphaZero paper](https://arxiv.org/abs/1712.01815)
* [Simple AlphaZero implementation](https://web.stanford.edu/~surag/posts/alphazero.html)
* [Implementing tensor/scalar representations of boards and actions](https://github.com/iamlucaswolf/gym-chess)
* [ChatGPT](https://chat.openai.com/chat)
