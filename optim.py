# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 10:49:47 2016

@author: jrbtaylor
"""

from __future__ import print_function

import numpy
import theano
from theano import tensor as T

from collections import OrderedDict
import timeit
import math

def sgd(data, model, cost, x, y,
        n_train_batches, n_valid_batches, n_test_batches,
        batch_size, learning_rate, momentum,
        init_patience, n_epochs):
    # Unpack the data
    train_set_x, train_set_y = data[0]
    valid_set_x, valid_set_y = data[1]
    test_set_x, test_set_y = data[2]
    
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

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
