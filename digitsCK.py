#!/usr/bin/python3


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import sys
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy




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


# *~*~*~*~*~*~*~*~*~*~* prepare the data ~*~*~*~*~*~*~*~*~*~*~*

# number of features
n = X_trn.shape[1]

# number of digits to identify
v = 10

# Turn each value ydigi into a one-dimensional vector of length v
y_trn = np.zeros((m0,v))
for i,yt in enumerate(ydigi):
    y_trn[i,yt] = 1

# Convert X_trn from a DataFrame to a numpy array
X_trn = np.asarray(X_trn)

# Convert training set from 1D array into 2D images to be fed to Conv2D
X_urn = np.zeros((m0, 28, 28, 1))
for i in range(m0):
    X_urn[i,:,:,0] = X_trn[i,:].reshape(28,28)

# Convert testing set from  1D array into 2D images to be used for prediction
X_ust = np.zeros((m1, 28, 28, 1))
for i in range(m1):
    X_ust[i,:,:,0] = X_tst[i,:].reshape(28,28)

# *~*~*~*~*~*~*~*~*~*~* prepare the data ~*~*~*~*~*~*~*~*~*~*~*


# *~*~*~*~*~*~*~*~*~*~* Keras Sequential ~*~*~*~*~*~*~*~*~*~*~*

CK_model = Sequential([
            Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu'), 
            MaxPooling2D(pool_size=(2,2)), 
            Flatten(), 
            Dense(100, activation='sigmoid'),
            Dense(10, activation='softmax')
            ])
CK_model.compile(SGD(lr=0.03), loss='categorical_crossentropy', metrics=['accuracy'])

hst = CK_model.fit(X_urn, y_trn, batch_size=32, epochs=100, verbose=1, validation_split=0.20)

# *~*~*~*~*~*~*~*~*~*~* Keras Sequential ~*~*~*~*~*~*~*~*~*~*~*



# *~*~*~*~*~*~*~*~*~*~* Accurcy and Loss ~*~*~*~*~*~*~*~*~*~*~*

plt.figure(figsize=(12,5))
ax = plt.subplot(1, 2, 1)
plt.plot(hst.history['acc']);  plt.plot(hst.history['val_acc']);  plt.legend(['train','validation']);
plt.xlabel('epochs');  plt.ylabel('accuracy');  plt.title('model accuracy')

plt.subplot(1,2,2)
plt.plot(hst.history['loss']);  plt.plot(hst.history['val_loss']);  plt.legend(['train','validation']);
plt.xlabel('epochs');  plt.ylabel('loss');  plt.title('model loss')

plt.show()

# *~*~*~*~*~*~*~*~*~*~* Accurcy and Loss ~*~*~*~*~*~*~*~*~*~*~*


# *~*~*~*~*~*~*~*~*~*~* Accurcy and Loss ~*~*~*~*~*~*~*~*~*~*~*

plt.figure(figsize=(12,5))
ax = plt.subplot(1, 2, 1)
plt.plot(hst.history['acc']);  plt.plot(hst.history['val_acc']);  plt.legend(['train','validation']);
plt.xlabel('epochs');  plt.ylabel('accuracy');  plt.title('model accuracy')

plt.subplot(1,2,2)
plt.plot(hst.history['loss']);  plt.plot(hst.history['val_loss']);  plt.legend(['train','validation']);
plt.xlabel('epochs');  plt.ylabel('loss');  plt.title('model loss')

plt.show()

# *~*~*~*~*~*~*~*~*~*~* Accurcy and Loss ~*~*~*~*~*~*~*~*~*~*~*


# *~*~*~*~*~*~*~*~*~*~*~*~* Output ~*~*~*~*~*~*~*~*~*~*~*~*~*~*

# Predict the classes for test set
y_prd = CK_model.predict(X_ust)

outdig = [np.argmax(c0) for c0 in y_prd]

outcsv = pd.DataFrame({'ImageId':np.arange(m1)+1, 'Label':outdig})
outcsv.to_csv('output.csv', index=False)

# *~*~*~*~*~*~*~*~*~*~*~*~* Output ~*~*~*~*~*~*~*~*~*~*~*~*~*~*

