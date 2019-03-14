#!/usr/bin/python3


import sys
#import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from numpy import random



### *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
### *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*


if (sys.platform == 'linux'):
    thap = '/media/anand/Chiba/Users/Anand/Documents/home/' + \
                'python/kaggle/DigitRecognizer'
else:
    thap = 'C:/Users/Anand/Documents/home/python/kaggle/DigitRecognizer'
#endelse
os.chdir(thap)

# The files for test and train
f1 = 'train.csv'
g1 = 'test.csv'

# Read the csv file into a DataFrame. Remove nrows argument to read full set
df0 = pd.read_csv(f1, sep=',', nrows=500)
dg0 = pd.read_csv(g1, sep=',', nrows=500)

# Extract columns from df0 to form X_trn and y_trn
X_trn = df0.filter(regex='pixel*', axis=1)
y_trn = df0['label']
# X_trn = df0.drop('label', axis=1)

# Size of df0
sz_df0 = df0.shape

# The whole of dg0 is X_tst
X_tst = dg0.copy()

# Number of digits to display in each row and each column
gs = 10

# To plot the digits in a grid of of size gs x gs
fig = plt.figure()
fig.set_size_inches(8, 8)
#fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, \
            wspace=0.005, hspace=0.005)

for i in range(gs**2):
    # Extract a random digit from X_trn
    a1 = X_trn.iloc[int(random.rand()*sz_df0[0]-1)].values.reshape(28,28)

    # Sequentially view numbers from X_tst
    a1 = X_tst.iloc[i].values.reshape(28,28)

    # Make a subplot, set ticklabels to '', set tick lengths to 0
    ax = fig.add_subplot(gs, gs, i+1)
    ax.set_xticklabels('');  ax.tick_params(axis='x', length=0);
    ax.set_yticklabels('');  ax.tick_params(axis='y', length=0);
    
    # Display the numpy array as an image
    plt.imshow(a1)

#    a0 = X_trn.iloc[int(random.rand()*sz_df0[0]-1)]
#    a2 = Image.fromarray((a1*255).astype(np.uint8), mode='L')
    # This generates a single 28 x 28 window for each a2
#    a2.show()


### *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
### *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
