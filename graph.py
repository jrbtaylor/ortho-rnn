# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 21:56:03 2016

@author: jrbtaylor
"""
#%%
import numpy
import csv
import matplotlib
import matplotlib.pyplot as plt

# what will eventually be the function input when I finish this:
filename = '/home/ml/jtaylo55/Documents/code/ortho-rnn/orthopen3.csv'
n_in = 4
n_hidden = 64
metric = 'Training loss'

# open the file
file = open(filename,'r')
reader = csv.reader(file,delimiter=',')

# read the labels from the first row
categories = reader.next()
assert(categories==['Hyperparameters', 'n_in', 'n_hidden', 'Training loss',
                    'Training error%', 'Validation loss', 'Validation error%',
                    'Test error%', '|Wx|', '|Wh|', '|Wy|', '|WWt-I|'])

# read the desired metric
col = categories.index(metric)
data = []
hyperparam = []
for row in reader:
    if row[categories.index('n_in')] == str(n_in) \
    and row[categories.index('n_hidden')] == str(n_hidden):
        # note: data is stored in string like "['2.34', '3.45']"
        datastring = row[col]
        datastring = datastring[1:-1] # strip '[' and ']'
        datastring = datastring.split(', ') # split into numbers
        data.append([float(d) for d in datastring])
        hyperparam.append(row[0])

# plot it
for idx,d in enumerate(data):
    plt.plot(range(1,len(d)+1),d,label=hyperparam[idx])
plt.xlabel('Epoch')
plt.ylabel(metric)
plt.title(metric+' vs hyperparameter'+
          '\n for n_in='+str(n_in)+', n_hidden='+str(n_hidden))

ymin,ymax = plt.ylim()
#plt.ylim(0,ymax)
plt.legend()

plt.show()
























