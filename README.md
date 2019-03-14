## Neural Network to recognize hand-written digits

This exercise is the solution to kaggle competition to recognize the handwritten digits from MNIST database. The train.csv file contains 42,000 labelled examples, while the test.csv contains 18,000 unlabelled examples.

File display_digits.py displays the train or test examples in a 10 x 10 grid.

The solution is implemented in two ways:

### 1. Artificial Neural Network from scratch without using any library or framework

The file digitsNN.py implements an ANN from scratch. The forward propagation, cost function and back propagation was computed without use of any ML/DL library or framework. The purpose of this was to understand first-hand the workings of an ANN, rather than to achieve a high accuracy score.

### 2. ANN using Keras framework

This will follow soon.