# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 21:56:03 2016

@author: jrbtaylor
"""
#%%
import os
import numpy
import csv
import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.
import matplotlib.pyplot as plt

def make_graph(filename,n_in,n_hidden,metric):
    # open the file
    if not filename[-4:]=='.csv':
        filename = filename+'.csv'
    file = open(filename,'r')
    reader = csv.reader(file,delimiter=',')
    
    # read the labels from the first row
    categories = reader.next()
    assert(categories==['Hyperparameters', 'n_in', 'n_hidden', 'Training loss',
                        'Training error', 'Validation loss', 'Validation error',
                        'Test error', '|Wx|', '|Wh|', '|Wy|', '|WWt-I|'])
    assert(metric in categories)
    
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
    if len(data)>0: # don't want to save blank graphs
        
        # loop through each row and plot
        for idx,d in enumerate(data):
            plt.plot(range(1,len(d)+1),d,label=hyperparam[idx])
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(metric+' vs hyperparameter'+
                  '\n for n_in='+str(n_in)+', n_hidden='+str(n_hidden))
        
        # basic outlier detection for setting graph limits
        ymaxs = [numpy.max(d) for d in data]
        ylim = numpy.max(ymaxs)
        everythingelse = [ymaxs[i] for i in range(len(ymaxs)) \
                            if i!=numpy.argmax(ymaxs)]
        if ylim > 10*numpy.max(everythingelse):
            ylim = numpy.max(everythingelse)
        plt.ylim(0,ylim)
        plt.legend()
        
        # save it
        subfolder = os.path.join(os.getcwd(),'results')
        if not os.path.isdir(subfolder):
            os.mkdir(subfolder)
        subfolder = os.path.join(subfolder,filename.split('.')[0])
        if not os.path.isdir(subfolder):
            os.mkdir(subfolder)
        saveto = metric+'_'+str(n_in)+'in_'+str(n_hidden)+'hidden'
        plt.savefig(os.path.join(subfolder,saveto+'.png'))
        plt.close()
    else:
        print('Empty graph not saved for '+filename+', '+metric+', '
              +str(n_in)+'in, '+str(n_hidden)+'hidden')


def make_all(filename):
    import itertools
    
    # open the file
    if not filename[-4:]=='.csv':
        filename = filename+'.csv'
    file = open(filename,'r')
    reader = csv.reader(file,delimiter=',')
    
    # read the labels from the first row
    categories = reader.next()
    assert(categories==['Hyperparameters', 'n_in', 'n_hidden', 'Training loss',
                        'Training error', 'Validation loss', 'Validation error',
                        'Test error', '|Wx|', '|Wh|', '|Wy|', '|WWt-I|'])
    
    # read the file
    n_ins = []
    n_hiddens = []
    metrics = categories[3:]
    metrics.remove('Test error')
    for row in reader:
        n_ins.append(row[categories.index('n_in')])
        n_hiddens.append(row[categories.index('n_hidden')])
    n_ins = list(set(n_ins))
    n_hiddens = list(set(n_hiddens))

    # loop through everything and make all the graphs
    for n_in,n_hidden,metric in itertools.product(n_ins,n_hiddens,metrics):
        print(('making graph for '+metric+' : n_in = %s, n_hidden = %s') \
              % (n_in,n_hidden))
        make_graph(filename,n_in,n_hidden,metric)



















