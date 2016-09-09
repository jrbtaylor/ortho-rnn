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
import graph


def experiment(learning_rate=1e-1, lr_decay=0.99, bptt_limit=784, momentum=0.9,
               l1_reg=0, l2_reg=1e-3, n_epochs=500, init_patience=500, 
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
                         learning_rate,lr_decay,momentum,
                         init_patience,n_epochs)
    
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
                         learning_rate,lr_decay,momentum,
                         init_patience,n_epochs)
    
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
                         learning_rate,lr_decay,momentum,
                         init_patience,n_epochs)
        
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
                         learning_rate,lr_decay,momentum,
                         init_patience,n_epochs)
    
    # repeat the test and log all the returns
    def repeat_test(fn,rep):
        best_test = numpy.inf
        for n in range(rep):
            train_loss,train_errors, \
            validation_loss,validation_errors,test_score, monitors = fn(n)
            if test_score<best_test:
                best_test = test_score
                train_loss_best = train_loss
                train_errors_best = train_errors
                validation_loss_best = validation_loss
                validation_errors_best = validation_errors
                test_score_best = test_score
                monitors_best = monitors
        return ([train_loss_best,train_errors_best],[validation_loss_best,
                validation_errors_best],test_score_best,monitors_best)
    
    # log results in a csv file
    def log_results(filename,line,hyperparam,n_in,n_hidden,
                    trn,vld,tst,mntrs):
        import csv
        import os
        # unpack the results
        trn_loss = trn[0]
        trn_error = trn[1]
        vld_loss = vld[0]
        vld_error = vld[1]
        normWx = [n[0] for n in mntrs]
        normWh = [n[1] for n in mntrs]
        normWy = [n[2] for n in mntrs]
        orthog = [n[3] for n in mntrs]
        if line==0:
            # check if old log exists and delete
            if os.path.isfile(filename):
                os.remove(filename)
        file = open(filename,'a')
        writer = csv.writer(file)
        if line==0:
            writer.writerow(('Hyperparameters','n_in','n_hidden',
                             'Training loss','Training error%',
                             'Validation loss','Validation error%',
                             'Test error%',
                             '|Wx|','|Wh|','|Wy|','|WWt-I|'))
        writer.writerow((hyperparam,n_in,n_hidden,
                         trn_loss,trn_error,
                         vld_loss,vld_error,tst,
                         normWx,normWh,normWy,orthog))
    
    # experiment parameters
    n_ins = [7,16]
    n_hiddens = [64,256]
    
    # hyperparameter search for gradient clipping
    if any('gradclip' in s for s in to_run):
        clip_limits = [10**x for x in range(-3,2)]
        for idx,(clip_limit,n_in,n_hidden) in \
                enumerate(itertools.product(clip_limits,
                                            n_ins,
                                            n_hiddens)):
            fn = lambda seed: test_gradclip(clip_limit,n_in,n_hidden,seed)
            trn,vld,tst,mntrs = repeat_test(fn,repeated_exp)
            log_results('gradclip.csv',idx,clip_limit,n_in,n_hidden,
                        trn,vld,tst,mntrs)
        graph.make_all('gradclip.csv')
        
    # hyperparameter search for orthogonality penalty
    if any('ortho1' in s for s in to_run):
        eps_idems = [10**x for x in range(-4,0)]
        eps_norms = [10**x for x in range(-4,0)]
        for idx,(eps_idem,eps_norm,n_in,n_hidden) in \
                enumerate(itertools.product(eps_idems,
                                            eps_norms,
                                            n_ins,
                                            n_hiddens)):
            fn = lambda seed: test_orthopen(eps_idem,eps_norm,n_in,n_hidden,seed)
            trn,vld,tst,mntrs = repeat_test(fn,repeated_exp)
            log_results('orthopen1.csv',idx,[eps_idem,eps_norm],n_in,n_hidden,
                        trn,vld,tst,mntrs)
        graph.make_all('orthopen1.csv')
        
    # hyperparameter search for orthogonality penalty
    if any('ortho2' in s for s in to_run):
        eps_idems = [10**x for x in range(-4,0)]
        eps_norms = [10**x for x in range(-4,0)]
        for idx,(eps_idem,eps_norm,n_in,n_hidden) in \
                enumerate(itertools.product(eps_idems,
                                            eps_norms,
                                            n_ins,
                                            n_hiddens)):
            fn = lambda seed: test_orthopen2(eps_idem,eps_norm,n_in,n_hidden,seed)
            trn,vld,tst,mntrs = repeat_test(fn,repeated_exp)
            log_results('orthopen2.csv',idx,[eps_idem,eps_norm],n_in,n_hidden,
                        trn,vld,tst,mntrs)
        graph.make_all('orthopen2.csv')
    
    # hyperparameter search for orthogonality penalty
    if any('ortho3' in s for s in to_run):
        eps_orthos = [10**x for x in range(-5,0)]
        for idx,(eps_ortho,n_in,n_hidden) in \
                enumerate(itertools.product(eps_orthos,
                                            n_ins,
                                            n_hiddens)):
            fn = lambda seed: test_orthopen3(eps_ortho,n_in,n_hidden,seed)
            trn,vld,tst,mntrs = repeat_test(fn,repeated_exp)
            log_results('orthopen3.csv',idx,eps_ortho,n_in,n_hidden,
                        trn,vld,tst,mntrs)
        graph.make_all('orthopen3.csv')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run RNN experiments')
    parser.add_argument('--exp',nargs='*',type=str,
                        default=['gradclip','ortho1','ortho2','ortho3'])
    print(parser.parse_args().exp)
    experiment(to_run=parser.parse_args().exp)