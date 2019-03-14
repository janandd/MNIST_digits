#!/usr/bin/python3


import sys
import numpy as np
import pandas as pd
import os
from math import exp
from math import sqrt
import random
from time import time



# The sigmoid function
def sigmoid(axe):
    return (1 / (1 + exp(-axe)))


# Random initialization of weights, Thetai
def rand_initialize(Tsz0, Tsz1):
    # Tsz0 & Tsz1 are respectively lengths of dimensions 0 & 1 of Thetai
    # Thetai is initialized to a random number in range [-epsilon,epsilon]
    epsilon = sqrt(6) / sqrt(Tsz0 + Tsz1)
    return np.asarray([[random.uniform(-epsilon,epsilon) for j in range(Tsz1)] \
                for i in range (Tsz0)])


def forw_propagate(X_dig):
    # Get the hidden units
    lyr2 = np.asarray([sigmoid(c0) for c0 in np.matmul(Theta1,X_dig)])
    lyr2 = np.insert(lyr2, 0, 1)

    # Final output layer.
    lyr3 = np.asarray([sigmoid(c0) for c0 in np.matmul(Theta2,lyr2)])

    return lyr2, lyr3


### *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
### *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*


# *~*~*~*~*~*~*~*~*~*~*~* read the data *~*~*~*~*~*~*~*~*~*~*~*

if (sys.platform == 'linux'):
    thap = '/media/anand/Chiba/Users/Anand/Documents/home/' + \
                'python/kaggle/DigitRecognizer'
else:
    thap = 'C:/Users/Anand/Documents/home/python/kaggle/DigitRecognizer'
os.chdir(thap)

# The files for test and train
f1 = 'train.csv'
g1 = 'test.csv'

# No. of examples to read for a start
m = 500            # no. of examples to read

# Read the csv files into DataFrames. Remove nrows argument to read full set
df0 = pd.read_csv(f1, sep=',')            # , nrows=m
dg0 = pd.read_csv(g1, sep=',')            # , nrows=m

# Number of training and testing examples
m0 = df0.shape[0]        # number of training examples
m1 = dg0.shape[0]        # number of testing examples

# Extract columns from df0 to form training set, X_trn and y_trn
X_trn = df0.drop('label', axis=1)
ydigi = df0['label']
#X_trn = df0.filter(regex='pixel*', axis=1)

# The whole of dg0 is testing set, X_tst, but as a numpy array
X_tst = np.asarray(dg0.copy())

# *~*~*~*~*~*~*~*~*~*~*~* read the data *~*~*~*~*~*~*~*~*~*~*~*


# *~*~*~*~*~* Initialize artificial neural network ~*~*~*~*~*~*

# We use a single hidden layer to start with

# Size of df0
n = X_trn.shape[1]        # no. of features

# no. of units in the (single) hidden layer
l = 400

# no. of digits to identify
v = 10

# Turn each value ydigi into a one-dimensional vector of length v
y_trn = np.zeros((m0,v))
for i,yt in enumerate(ydigi):
    y_trn[i,yt] = 1

# Insert bias column at the start of X_trn
X_trn.insert(loc=0, column='bias', value=[1 for i in range(m0)])
X_trn = np.asarray(X_trn)

# numpy arrays to hold the weights
# dimension of Thetai is length of layer i+1 x length of layer i
# Therefore, Theta1 is (l,n+1) and Theta2 is (v,l+1)
# Final layer has v units, since this is a multi-class classification problem
Theta1 = rand_initialize(l, n+1)
Theta2 = rand_initialize(v, l+1)

# Reguarization parameter, lambda; had to use another name for obvious reasons
dambla = 0.09

# the learning rate for Gradient Descent
# It turns out, a larger alpha results in a "OverflowError: math range error"
alpha = 0.03

# no. of iterations
iters = 100

# *~*~*~*~*~* Initialize artificial neural network ~*~*~*~*~*~*

# To hold partial derivative of cost function for each Thetai
# Therefore, dimensions of D2_lij are same as Theta2
# similarly, dimensions of D1_lij are same as Theta1
D2_lij = np.zeros((v,l+1))
D1_lij = np.zeros((l,n+1))

# Array to hold the Cost Function at each iteration
coster = np.zeros(iters)

# Flag to check reduction in cost function
fl = np.zeros(iters).astype('int16')

# Start time at start of iterations
t0 = time()

for i in range(iters):
    # Deltai - accumulator of values deli at each iteration
    Delta2 = np.zeros((v,l+1))
    Delta1 = np.zeros((l,n+1))

    # To store the sum of residual like term from cost function
    A3sum = 0.0

    for j in range(m0):
        # *~*~*~*~*~*~*~*~*~* Forward Propagation *~*~*~*~*~*~*~*~*~*~*

        A2, A3 = forw_propagate(X_trn[j,:])

        # *~*~*~*~*~*~*~*~*~* Forward Propagation *~*~*~*~*~*~*~*~*~*~*

        # *~*~*~*~*~*~*~*~*~ Gradient Computation *~*~*~*~*~*~*~*~*~*~*
    
        # delta for 3rd layer (output layer) is simply the difference
        del3 = A3 - y_trn[j,:]
    
        # delta for 2nd layer (hidden layer) is given below
        A2der = A2 * (1-A2)
        del2 = np.matmul(del3, Theta2) * A2der
    
        # Accumulators of the partial derivatives
        Delta2 += np.dot(del3.reshape(v,1), A2.reshape(1,l+1))
        Delta1 += np.dot(del2[1:].reshape(l,1), X_trn[j,:].reshape(1,n+1))

        # First term of the cost function
        A3sum += np.sum([c0*np.log(d0) + (1-c0)*np.log(1-d0) for \
                    c0,d0 in zip(y_trn[j,:],A3)])

        # *~*~*~*~*~*~*~*~*~* Gradient Computation*~*~*~*~*~*~*~*~*~*~*


    # *~*~*~*~*~*~*~*~*~*~*~ Cost Function ~*~*~*~*~*~*~*~*~*~*~*~*

    # Regularization of the cost function
    ThSqS = np.sum(np.sum(Theta1[:,1:] ** 2)) + np.sum(np.sum(Theta2[:,1:] ** 2))

    # The cost function
    coster[i] = -1/m0 * A3sum + dambla/(2*m0) * ThSqS

    # *~*~*~*~*~*~*~*~*~*~*~ Cost Function ~*~*~*~*~*~*~*~*~*~*~*~*

    # *~*~*~*~*~*~*~*~*~*~ Gradient Descent *~*~*~*~*~*~*~*~*~*~*~*

    D2_lij[:,0] = 1/m0 * Delta2[:,0]
    D2_lij[:,1:] = 1/m0 * Delta2[:,1:] + dambla * Theta2[:,1:]

    D1_lij[:,0] = 1/m0 * Delta1[:,0]
    D1_lij[:,1:] = 1/m0 * Delta1[:,1:] + dambla * Theta1[:,1:]

    Theta1 -= np.multiply(alpha, D1_lij)
    Theta2 -= np.multiply(alpha, D2_lij)

    # *~*~*~*~*~*~*~*~*~*~ Gradient Descent *~*~*~*~*~*~*~*~*~*~*~*

    # *~*~*~*~*~*~*~*~*~*~*~ flag checking ~*~*~*~*~*~*~*~*~*~*~*~*

    # Set flag 1 when cost function reduces by more than 1e-5
    if (coster[i]-coster[i-1] < -1e-4):
        fl[i] = 1

    # Stop iterating if flag is 1 in at least 20 of the last 50 times
    if ((i > 51) and (np.sum(fl[i-49:i+1]) > 20)):
        break

    # *~*~*~*~*~*~*~*~*~*~*~ flag checking ~*~*~*~*~*~*~*~*~*~*~*~*

    # Progress of the iteration loop
    if (i%100 == 0):
        print(i, coster[i])


# Print the time required to complete all the iterations
print(time() - t0)


# *~*~*~*~*~*~*~*~*~*~*~* For the test ~*~*~*~*~*~*~*~*~*~*~*~*

outdig = []

for k in range(m1):
    # Get the hidden units
    B2 = np.asarray([sigmoid(c0) for c0 in np.matmul(Theta1[:,1:],X_tst[k,:])])

    # Final output layer.
    B3 = np.asarray([sigmoid(c0) for c0 in np.matmul(Theta2[:,1:],B2)])

    outdig.append(np.argmax(B3))

# *~*~*~*~*~*~*~*~*~*~*~* For the test ~*~*~*~*~*~*~*~*~*~*~*~*


# *~*~*~*~*~*~*~*~*~*~*~*~ Output csv *~*~*~*~*~*~*~*~*~*~*~*~*

outcsv = pd.DataFrame({'ImageId':np.arange(m1)+1, 'Label':outdig})
#outcsv.to_csv('output.csv', index=False)

# *~*~*~*~*~*~*~*~*~*~*~*~ Output csv *~*~*~*~*~*~*~*~*~*~*~*~*

### *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
### *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*


""" *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
# Forward Propagation for a single example
# numpy array to hold units in the hidden layer
A2 = np.ndarray(l+1)

# Insert bias column at the start of X_trn
X_trn.insert(loc=0, column='bias', value=[1 for i in range(m0)])

# First try with a single random example
Xi = X_trn.iloc[345,:]

# Get the hidden units
A2 = [sigmoid(c0) for c0 in np.matmul(Theta1,Xi)]
A2 = np.insert(A2, 0, 1)

# Final output layer.
A3 = np.asarray([sigmoid(c0) for c0 in np.matmul(Theta2,A2)])
*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~* """


""" *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
# Random initialization for checking
for i in range(l):
    for j in range(n+1):
        # Doesn't work without integers due to rounding errors
        Theta1[i,j] = int(random.random()*100)

# The correct way to perform matrix multiplication
# verified by calculating the same in for loops
# and list comprehension
A2x = np.zeros(l)
for j in range(l):
    for k in range(n+1):
        A2x[j] += Theta1[j,k] * Xi[k]

A2y = np.asarray([sum(Theta1[j,k]*Xi[k] for k in range(n+1)) for j in range(l)])

A2b = np.dot(Theta1, Xi);  A2c = np.matmul(Theta1, Xi);

# In this case, A2b == A2c == A2y == A2x
*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~* """


""" *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
Theta1 = np.zeros((l,n+1))
Theta2 = np.zeros((v,l+1))

# Random initialization of weights for checking
for i in range(l):
    for j in range(n+1):
        Theta1[i,j] = random.random()
        # Doesn't work without integers due to rounding errors
#        Theta1[i,j] = random.randint(1,100)
   
for i in range(v):
    for j in range(l+1):
        Theta2[i,j] = random.random()
#        Theta2[i,j] = random.randint(1,100)
*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~* """


""" *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
# numpy array to hold units in the hidden layer
A2 = np.zeros((m0,l+1))

# Final output layer
A3 = np.zeros((m0,v))

    # Residual like [?] term from cost function
    A3int = np.asarray([[c1*np.log(d1) + (1-c1)*np.log(1-d1) for c1,d1 in zip(c0,d0)] \
                for c0,d0 in zip(y_trn,A3)])

    # The cost function
    coster[i] = -1/m0 * sum(sum(A3int)) + dambla/(2*m0) * ThSqS
*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~* """


""" *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
    delta2 = np.matmul(delta3, Theta2[:,1:]) * A2deriv

    b1 = np.zeros((m0, l))
    for j in range(m0):
        for k in range(l):
            for p in range(v):
                b1[j,k] += del3[j,p] * (Theta2[:,1:])[p,k]
    # delta2 == b1 is True
*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~* """


""" *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
    #                   [?] IMPORTANT [?]
    # Cost Function, error, and backward propagation has to be
    # calculated for each example separately during each
    # iteration, and not once you are done computing the output
    # values for all the examples together in one iteration.
    
    # *~*~*~*~*~*~*~*~*~* Gradient Computation*~*~*~*~*~*~*~*~*~*~*

    # delta for 3rd layer (output layer) is simply the difference
    delta3 = A3 - y_trn

    # delta for 2nd layer (hidden layer) is given below
    A2deriv = A2[:,1:] * (1-A2[:,1:])    
    delta2 = np.matmul(delta3, Theta2[:,1:]) * A2deriv

    # *~*~*~*~*~*~*~*~*~* Gradient Computation*~*~*~*~*~*~*~*~*~*~*
*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~* """
