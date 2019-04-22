## Neural Network to recognize hand-written digits

This exercise is the solution to kaggle competition to recognize the handwritten digits from MNIST database. The train.csv file contains 42,000 labelled examples, while the test.csv contains 18,000 unlabelled examples.

File display_digits.py displays the train or test examples in a 10 x 10 grid.

The solution is implemented in two ways:

### 1. digits_NN.py
Artificial Neural Network from scratch without using any library or framework
The file digitsNN.py implements an ANN from scratch. The forward propagation, cost function and back propagation was computed without use of any ML/DL library or framework. The purpose of this was to understand first-hand the workings of an ANN, rather than to achieve a high accuracy score.

### 2. digitsNK.py
Artificial Neural Network using Keras framework
We use a single Dense layer with 600 neurons, thus basically replicating the ANN from file above. We observe a vast improvement in both accuracy and speed.

### 3. digitsCK.py
A Convolutional Neural Network using Keras
The first layer is a Conv2D layer with 32 filters and 3X3 kernel size. This is followed by MaxPooling2D, flattening, and a Dense layer with 100 neurons. The final output layer is another Dense layer with 10 outputs. Activation functions are chosen after trial and error, as most other combinations either failed to converge completely or were worse than the current combination. The accuracy now is at 98.14% using the CNN.


