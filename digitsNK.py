#!/usr/bin/python3


import sys
import numpy as np
import pandas as pd
import os
from time import time

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD


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

# The whole of dg0 is testing set, X_tst, but as a numpy array
X_tst = np.asarray(dg0.copy())

# *~*~*~*~*~*~*~*~*~*~*~* read the data *~*~*~*~*~*~*~*~*~*~*~*


# *~*~*~*~*~* Initialize artificial neural network ~*~*~*~*~*~*

# We use a single hidden layer to start with

# Size of df0
n = X_trn.shape[1]        # no. of features

# no. of units in the (single) hidden layer
l2 = 600

# no. of digits to identify
v = 10

# Turn each value ydigi into a one-dimensional vector of length v
y_trn = np.zeros((m0,v))
for i,yt in enumerate(ydigi):
    y_trn[i,yt] = 1

# Change DataFrame to a numpy array
X_trn = np.asarray(X_trn)

# Reguarization parameter, lambda; had to use another name for obvious reasons
dambla = 0.09

# the learning rate for Gradient Descent
# It turns out, a larger alpha results in a "OverflowError: math range error"
alpha = 0.03

# no. of iterations
iters = 4

# *~*~*~*~*~* Initialize artificial neural network ~*~*~*~*~*~*


# *~*~*~*~*~*~*~*~*~*~* keras Sequential ~*~*~*~*~*~*~*~*~*~*~*

# Start time at start of iterations
t0 = time()

#ledom = Sequential([
#            Dense(l2, input_shape=(n,), activation='sigmoid'), 
#            Dense(10, activation='sigmoid')
#            ])
#ledom.compile(SGD(lr=alpha), loss='mean_squared_error', metrics=['accuracy'])
#ledom.fit(X_trn, y_trn, batch_size=10, epochs=iters, verbose=1, 
#            validation_split=0.20)
#
#
#y_prd = ledom.predict(X_tst)

# Print the time required to complete all the iterations
#print(time() - t0)

# *~*~*~*~*~*~*~*~*~*~* keras Sequential ~*~*~*~*~*~*~*~*~*~*~*



# *~*~*~*~*~*~*~*~*~*~*~*~ Output csv *~*~*~*~*~*~*~*~*~*~*~*~*

#outdig = [np.argmax(c0) for c0 in y_prd]
#
#outcsv = pd.DataFrame({'ImageId':np.arange(m1)+1, 'Label':outdig})
#outcsv.to_csv('output.csv', index=False)

# *~*~*~*~*~*~*~*~*~*~*~*~ Output csv *~*~*~*~*~*~*~*~*~*~*~*~*