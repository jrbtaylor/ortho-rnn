# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 01:52:13 2016

@author: jrbtaylor
"""

from __future__ import print_function

import numpy
import theano
from theano import tensor as T

import itertools

import data
import recurrent
import optim


def experiment(learning_rate=1e-1, bptt_limit=784, momentum=0.9,
               l1_reg=0, l2_reg=1e-3, n_epochs=500, init_patience=20, 
               batch_size=1000, repeated_exp=10,
               to_run=['gradclip','ortho1','ortho2','ortho3']):
    
    # Load the data
    datasets = data.load('mnist.pkl.gz')
    train_set_x, train_set_y = datasets[0] # x is 50000x784 (flattened 28x28)
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    # Calculate batch numbers
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]//batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]//batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]//batch_size
    
    # Reshape and wrap the data
    def dataprep(n_in):
        def xreshape(x):
            return x.reshape([x.shape[0],x.shape[1]/n_in,n_in])
        return [(xreshape(train_set_x),train_set_y),
                (xreshape(valid_set_x),valid_set_y),
                (xreshape(test_set_x),test_set_y)]
    
    # Define all the experiment functions
    def test_gradclip(clip_limit,n_in,n_hidden,seed=1):
        # Build the model
        print('... building the gradient-clipped RNN')
        x = T.tensor3('x')
        y = T.ivector('y') # labels are a 1D vector of integers
        rng = numpy.random.RandomState(seed)
        model = recurrent.rnn(x,n_in,n_hidden,10,bptt_limit,rng)
        unclipped_cost = model.crossentropy(y)+l1_reg*model.L1+l2_reg*model.L2
        cost = theano.gradient.grad_clip(unclipped_cost,-clip_limit,clip_limit)
        return optim.sgd(dataprep(n_in),model,cost,x,y,n_train_batches,
                         n_valid_batches,n_test_batches,batch_size,
                         learning_rate,momentum,init_patience,n_epochs)
    
    def test_orthopen(eps_idem,eps_norm,n_in,n_hidden,seed=1):
        # Build the model
        print('... building the orthogonality-constrained RNN')
        x = T.tensor3('x')
        y = T.ivector('y') # labels are a 1D vector of integers
        rng = numpy.random.RandomState(seed)
        model = recurrent.rnn_ortho(x,n_in,n_hidden,10,bptt_limit,rng)
        cost = model.crossentropy(y)+l1_reg*model.L1+l2_reg*model.L2+ \
               eps_idem*model.idem+eps_norm*model.norm
        return optim.sgd(dataprep(n_in),model,cost,x,y,n_train_batches,
                         n_valid_batches,n_test_batches,batch_size,
                         learning_rate,momentum,init_patience,n_epochs)
    
    def test_orthopen2(eps_idem,eps_norm,n_in,n_hidden,seed=1):
        # Build the model
        print('... building the 2nd orthogonality-constrained RNN')
        x = T.tensor3('x')
        y = T.ivector('y') # labels are a 1D vector of integers
        rng = numpy.random.RandomState(seed)
        model = recurrent.rnn_ortho2(x,n_in,n_hidden,10,bptt_limit,rng)
        cost = model.crossentropy(y)+l1_reg*model.L1+l2_reg*model.L2+ \
               eps_idem*model.idem+eps_norm*model.norm
        return optim.sgd(dataprep(n_in),model,cost,x,y,n_train_batches,
                         n_valid_batches,n_test_batches,batch_size,
                         learning_rate,momentum,init_patience,n_epochs)
        
    def test_orthopen3(eps_ortho,n_in,n_hidden,seed=1):
        # Build the model
        print('... building the 3rd orthogonality-constrained RNN')
        x = T.tensor3('x')
        y = T.ivector('y') # labels are a 1D vector of integers
        rng = numpy.random.RandomState(seed)
        model = recurrent.rnn_ortho3(x,n_in,n_hidden,10,bptt_limit,rng)
        cost = model.crossentropy(y)+l1_reg*model.L1+l2_reg*model.L2+ \
               eps_ortho*model.ortho
        return optim.sgd(dataprep(n_in),model,cost,x,y,n_train_batches,
                         n_valid_batches,n_test_batches,batch_size,
                         learning_rate,momentum,init_patience,n_epochs)
    
    # repeat the test and log all the returns
    def repeat_test(fn,rep):
        val_results = numpy.zeros((rep,),dtype=theano.config.floatX)
        test_results = numpy.zeros((rep,),dtype=theano.config.floatX)
        monitors = numpy.zeros((rep,4),dtype=theano.config.floatX)
        for n in range(rep):
            val_results[n], test_results[n], monitors[n] = fn(n)
        return (val_results,test_results,monitors)
    
    # log results in a csv file
    def log_results(filename,line,hyperparam,n_in,n_hidden,
                    valid_results,test_results,monitors):
        import csv
        import os
        if line==0:
            # check if old log exists and delete
            if os.path.isfile(filename):
                os.remove(filename)
        file = open(filename,'a')
        writer = csv.writer(file)
        if line==0:
            writer.writerow(('Hyperparameters','n_in','n_hidden',
                             'Valid mean','Valid min','Valid var',
                             'Test mean','Test min','Test var',
                             '|Wx|','|Wh|','|Wy|','|WWt-I|'))
        writer.writerow((hyperparam,n_in,n_hidden,
                         numpy.mean(valid_results),numpy.min(valid_results),
                         numpy.var(valid_results),numpy.mean(test_results),
                         numpy.min(test_results),numpy.var(test_results),
                         numpy.mean(monitors[:,0]),numpy.mean(monitors[:,1]),
                         numpy.mean(monitors[:,2]),numpy.mean(monitors[:,3])))
    
    # experiment parameters
    n_ins = [1,2,4,8,16]
    n_hiddens = [64,128,256,512]
    
    # hyperparameter search for gradient clipping
    if any('gradclip' in s for s in to_run):
        clip_limits = [10**x for x in range(-4,5)]
        for idx,(clip_limit,n_in,n_hidden) in \
                enumerate(itertools.product(clip_limits,
                                            n_ins,
                                            n_hiddens)):
            fn = lambda seed: test_gradclip(clip_limit,n_in,n_hidden,seed)
            valid_results, test_results, monitors = repeat_test(fn,repeated_exp)
            log_results('gradclip.csv',idx,clip_limit,n_in,n_hidden,
                        valid_results,test_results,monitors)
        
    # hyperparameter search for orthogonality penalty
    if any('ortho1' in s for s in to_run):
        eps_idems = [10**x for x in range(-6,3)]
        eps_norms = [10**x for x in range(-6,3)]
        for idx,(eps_idem,eps_norm,n_in,n_hidden) in \
                enumerate(itertools.product(eps_idems,
                                            eps_norms,
                                            n_ins,
                                            n_hiddens)):
            fn = lambda seed: test_orthopen(eps_idem,eps_norm,n_in,n_hidden,seed)
            valid_results, test_results = repeat_test(fn,repeated_exp)
            log_results('orthopen1.csv',idx,[eps_idem,eps_norm],n_in,n_hidden,
                        valid_results,test_results)
        
    # hyperparameter search for orthogonality penalty
    if any('ortho2' in s for s in to_run):
        eps_idems = [10**x for x in range(-6,3)]
        eps_norms = [10**x for x in range(-6,3)]
        for idx,(eps_idem,eps_norm,n_in,n_hidden) in \
                enumerate(itertools.product(eps_idems,
                                            eps_norms,
                                            n_ins,
                                            n_hiddens)):
            fn = lambda seed: test_orthopen2(eps_idem,eps_norm,n_in,n_hidden,seed)
            valid_results, test_results = repeat_test(fn,repeated_exp)
            log_results('orthopen2.csv',idx,[eps_idem,eps_norm],n_in,n_hidden,
                        valid_results,test_results)
    
    # hyperparameter search for orthogonality penalty
    if any('ortho3' in s for s in to_run):
        eps_orthos = [10**x for x in range(-7,2)]
        for idx,(eps_ortho,n_in,n_hidden) in \
                enumerate(itertools.product(eps_orthos,
                                            n_ins,
                                            n_hiddens)):
            fn = lambda seed: test_orthopen3(eps_ortho,n_in,n_hidden,seed)
            valid_results, test_results = repeat_test(fn,repeated_exp)
            log_results('orthopen3.csv',idx,eps_ortho,n_in,n_hidden,
                        valid_results,test_results)

if __name__ == "__main__":
    
    experiment()