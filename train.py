# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 01:52:13 2016

@author: jrbtaylor
"""

from __future__ import print_function

import numpy
import theano
from theano import tensor as T

from collections import OrderedDict
import timeit
import math

import data
import recurrent

def compare_models(learning_rate = 1e-1,n_in=14,n_hidden=256,
                   bptt_limit=784,momentum=0.9,l1_reg=0,l2_reg=0.001,
                   n_epochs=500,init_patience=20,batch_size=1000,
                   clip_limit=1.,eps_idem=0.01,eps_norm=0.01,eps_ortho=0.0001,
                   repeated_exp=10):
    """
    Test RNNs on sequential MNIST
    
    Compare the soft constraint on recurrent weight matrix orthogonality
    with standard gradient clipping.
    """
    # Load the data
    datasets = data.load('mnist.pkl.gz')
    train_set_x, train_set_y = datasets[0] # x is 50000x784 (flattened 28x28)
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
    
    # Reshape the data
    def xreshape(x): # unflattens if n_in>1
    	return x.reshape([x.shape[0],x.shape[1]/n_in,n_in])
    train_set_x = xreshape(train_set_x)
    valid_set_x = xreshape(valid_set_x)
    test_set_x = xreshape(test_set_x)
    
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.tensor3('x')
    y = T.ivector('y') # labels are a 1D vector of integers

    def test_gradclip(clip_limit,seed=1):
        # Build the model
        print('... building the gradient-clipped RNN')
        model = recurrent.rnn(x,n_in,n_hidden,10,bptt_limit,numpy.random.RandomState(seed))
        unclipped_cost = model.crossentropy(y)+l1_reg*model.L1+l2_reg*model.L2
        cost = theano.gradient.grad_clip(unclipped_cost,-clip_limit,clip_limit)
        return train_model(model,cost)
    
    def test_orthopen(eps_idem,eps_norm,seed=1):
        # Build the model
        print('... building the orthogonality-constrained RNN')
        model = recurrent.rnn_ortho(x,n_in,n_hidden,10,bptt_limit,numpy.random.RandomState(seed))
        cost = model.crossentropy(y)+l1_reg*model.L1+l2_reg*model.L2+ \
               eps_idem*model.idem+eps_norm*model.norm
        return train_model(model,cost)
    
    def test_orthopen2(eps_idem,eps_norm,seed=1):
        # Build the model
        print('... building the 2nd orthogonality-constrained RNN')
        model = recurrent.rnn_ortho2(x,n_in,n_hidden,10,bptt_limit,numpy.random.RandomState(seed))
        cost = model.crossentropy(y)+l1_reg*model.L1+l2_reg*model.L2+ \
               eps_idem*model.idem+eps_norm*model.norm
        return train_model(model,cost)
        
    def test_orthopen3(eps_ortho,seed=1):
        # Build the model
        print('... building the 3rd orthogonality-constrained RNN')
        model = recurrent.rnn_ortho3(x,n_in,n_hidden,10,bptt_limit,numpy.random.RandomState(seed))
        cost = model.crossentropy(y)+l1_reg*model.L1+l2_reg*model.L2+ \
               eps_ortho*model.ortho
        return train_model(model,cost)
        
    def train_model(model,cost):
        # SGD w/ momentum
        # Initialize momentum
        gparams_mom = []
        for param in model.params:
            gparam_mom = theano.shared(numpy.zeros(param.get_value(borrow=True).shape,
                                                   dtype=theano.config.floatX))
            gparams_mom.append(gparam_mom)
        # Setup backprop
        gparams = T.grad(cost,model.params)
        updates = OrderedDict()
        # Momentum update
        for gparam_mom, gparam in zip(gparams_mom,gparams):
            updates[gparam_mom] = momentum*gparam_mom-(1.-momentum)*learning_rate*gparam
        # Parameter update
        for param,gparam_mom in zip(model.params,gparams_mom):
            updates[param] = param+updates[gparam_mom]
    
        # Compiled functions
        train_model = theano.function(inputs=[index],
                                      outputs=(cost,model.errors(y)),
                                      updates=updates,
                                      givens={
                                          x: train_set_x[index*batch_size:(index+1)*batch_size],
                                          y: train_set_y[index*batch_size:(index+1)*batch_size]})
        test_model = theano.function(inputs=[index],
                                     outputs=model.errors(y),
                                     givens={
                                         x: test_set_x[index*batch_size:(index+1)*batch_size],
                                         y: test_set_y[index*batch_size:(index+1)*batch_size]})
        valid_model = theano.function(inputs=[index],
                                     outputs=model.errors(y),
                                     givens={
                                         x: valid_set_x[index*batch_size:(index+1)*batch_size],
                                         y: valid_set_y[index*batch_size:(index+1)*batch_size]})
        
        # Train the model
        print('... training the model')
        # early-stopping parameter
        patience = init_patience*n_train_batches
        
        validation_frequency = n_train_batches
        best_validation_loss = numpy.inf
        test_score = 0.
        
        done_looping = False
        epoch = 0
        iter = 0
        while (epoch < n_epochs) and (not done_looping):
            start_time = timeit.default_timer()
            start_iter = iter
            epoch += 1
            train_loss_epoch = 0
            train_errors_epoch = 0
            for minibatch_index in range(n_train_batches):
                train_loss_batch,train_errors_batch = train_model(minibatch_index)
                train_loss_epoch += train_loss_batch
                train_errors_epoch += train_errors_batch
    
                iter = (epoch - 1) * n_train_batches + minibatch_index
                if (iter + 1) % validation_frequency == 0:
                    validation_losses = [valid_model(i)
                                         for i in range(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                    print('   validation error: %f %%' %
                        (this_validation_loss * 100.)
                    )
    
                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        patience = init_patience*n_train_batches
                        best_validation_loss = this_validation_loss
    
                        test_losses = [test_model(i)
                                       for i in range(n_test_batches)]
                        test_score = numpy.mean(test_losses)
    
                        print(('      test error: %f %%') %
                            (test_score * 100.)
                        )
                    elif math.isnan(train_loss_epoch):
                        print('Stopping training due to NaNs')
                        done_looping = True
                        break
                    else:
                        patience -= n_train_batches
    
                if patience <= 0:
                    done_looping = True
                    break
            end_time = timeit.default_timer()
            print(
                (
                    'Epoch %i, training loss: %f, training error: %f %%, time per sample: %f ms'
                    ) %
                    (
                        epoch,
                        train_loss_epoch/n_train_batches,
                        train_errors_epoch/n_train_batches*100,
                        (end_time-start_time)/(iter-start_iter)/batch_size*1000
                    )
                )
        print(
            (
                'Optimization complete with best validation error of %f %%,'
                'with test error %f %%'
            )
            % (best_validation_loss * 100., test_score * 100.)
        )
        return (best_validation_loss*100,test_score*100)

    # repeat the test and log all the returns
    def repeat_test(fn,rep):
        val_results = numpy.zeros((rep,),dtype=theano.config.floatX)
        test_results = numpy.zeros((rep,),dtype=theano.config.floatX)
        for n in range(rep):
            val_results[n], test_results[n] = fn(n)
        return (val_results,test_results)
    
    gradclip_val, gradclip_test = repeat_test(lambda seed: test_gradclip(clip_limit,seed),repeated_exp)
    orthopen_val, orthopen_test = repeat_test(lambda seed: test_orthopen(eps_idem,eps_norm,seed),repeated_exp)
    orthopen2_val, orthopen2_test = repeat_test(lambda seed: test_orthopen2(eps_idem,eps_norm,seed),repeated_exp)
    orthopen3_val, orthopen3_test = repeat_test(lambda seed: test_orthopen3(eps_ortho,seed),repeated_exp)
    
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('  Error rates over '+str(repeated_exp)+' experiments')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('--- Gradient Clipping ---')
    print('    validation:  mean '+str(numpy.mean(gradclip_val))+'%% ... min '+str(numpy.min(gradclip_val))+'%%')
    print('    test:        mean '+str(numpy.mean(gradclip_test))+'%% ... min '+str(numpy.min(gradclip_test))+'%%')
    print('--- |h-hW| Penalty ---')
    print('    validation:  mean '+str(numpy.mean(orthopen_val))+'%% ... min '+str(numpy.min(orthopen_val))+'%%')
    print('    test:        mean '+str(numpy.mean(orthopen_test))+'%% ... min '+str(numpy.min(orthopen_test))+'%%')
    print('--- |hW-hWW| Penalty ---')
    print('    validation:  mean '+str(numpy.mean(orthopen2_val))+'%% ... min '+str(numpy.min(orthopen2_val))+'%%')
    print('    test:        mean '+str(numpy.mean(orthopen2_test))+'%% ... min '+str(numpy.min(orthopen2_test))+'%%')
    print("--- |W'W-I| Penalty ---")
    print('    validation:  mean '+str(numpy.mean(orthopen3_val))+'%% ... min '+str(numpy.min(orthopen3_val))+'%%')
    print('    test:        mean '+str(numpy.mean(orthopen3_test))+'%% ... min '+str(numpy.min(orthopen3_test))+'%%')
    

if __name__ == "__main__":
    compare_models()