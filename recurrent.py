# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 14:46:19 2016

@author: jrbtaylor
"""

import numpy

import theano
from theano import tensor as T
from theano.tensor import tanh
from theano.tensor.nnet import relu, sigmoid
from theano.tensor.nnet.nnet import softmax, categorical_crossentropy

rng = numpy.random.RandomState(1)

def _uniform_weight(n1,n2):
    limit = numpy.sqrt(6./(n1+n2))
    return theano.shared((rng.uniform(low=-limit,
                                      high=limit,
                                      size=(n1,n2))
                         ).astype(theano.config.floatX),
                         borrow=True)

def _ortho_weight(n):
    W = rng.randn(n, n)
    u, s, v = numpy.linalg.svd(W)
    return theano.shared(u.astype(theano.config.floatX),
                         borrow=True)

def _zero_bias(n):
    return theano.shared(numpy.zeros((n,),dtype=theano.config.floatX),
                         borrow=True)

class rnn(object):
    def __init__(self,x,n_in,n_hidden,n_out,bptt_limit):
        """
        Initialize a basic single-layer RNN
        
        x:    symbolic input tensor
        n_in:    input dimensionality
        n_hidden:    # of hidden units
        hidden_activation:    non-linearity at hidden units (e.g. relu)
        n_out:    # of output units
        output_activation:    non-linearity at output units (e.g. softmax)
        bptt_limit:    # of steps for backprop through time
        """
        self.Wx = _uniform_weight(n_in,n_hidden)
        self.Wh = _ortho_weight(n_hidden)
        self.Wy = _uniform_weight(n_hidden,n_out)
        self.bh = _zero_bias(n_hidden)
        self.by = _zero_bias(n_out)
        self.params = [self.Wx,self.Wh,self.Wy,self.bh,self.by]
        self.W = [self.Wx,self.Wy] # don't incl recurrent weights in decay
        self.L1 = numpy.sum([abs(w).sum() for w in self.W])
        self.L2 = numpy.sum([(w**2).sum() for w in self.W])
        
        def step(x_t,h_tm1,Wx,Wh,Wy,bh,by):
            h_t = relu(T.dot(x_t,Wx)+T.dot(h_tm1,Wh)+bh)
            y_t = softmax(T.dot(h_t,Wy)+by)
            return [h_t,y_t]
        h0 = T.zeros((n_hidden,),dtype=theano.config.floatX)
        ([h,y],_) = theano.scan(fn=step, 
                                sequences=x.dimshuffle([1,0,2]),
                                outputs_info=[T.alloc(h0,x.shape[0],n_hidden),
                                              None],
                                non_sequences=[self.Wx,self.Wh,self.Wy,
                                               self.bh,self.by],
                                strict=True)
        self.output = y[-1]
        self.pred = T.argmax(self.output,axis=-1)

    def crossentropy(self,y):
        return T.mean(categorical_crossentropy(self.output,y))
    
    def errors(self,y):
        return T.mean(T.neq(self.pred,y))
        

class rnn_ortho(object):
    def __init__(self,x,n_in,n_hidden,n_out,bptt_limit):
        """
        Initialize a basic single-layer RNN w/ soft orthogonality penalty
        
        x:    symbolic input tensor
        n_in:    input dimensionality
        n_hidden:    # of hidden units
        hidden_activation:    non-linearity at hidden units (e.g. relu)
        n_out:    # of output units
        output_activation:    non-linearity at output units (e.g. softmax)
        bptt_limit:    # of steps for backprop through time
        """
        self.Wx = _uniform_weight(n_in,n_hidden)
        self.Wh = _ortho_weight(n_hidden)
        self.Wy = _uniform_weight(n_hidden,n_out)
        self.bh = _zero_bias(n_hidden)
        self.by = _zero_bias(n_out)
        self.params = [self.Wx,self.Wh,self.Wy,self.bh,self.by]
        self.W = [self.Wx,self.Wy] # don't incl recurrent weights in decay
        self.L1 = numpy.sum([abs(w).sum() for w in self.W])
        self.L2 = numpy.sum([(w**2).sum() for w in self.W])
        
        def step(x_t,h_tm1,Wx,Wh,Wy,bh,by):
            hWh = T.dot(h_tm1,Wh)
            hmh = T.mean(T.sqr(h_tm1-hWh))
            h_t = relu(T.dot(x_t,Wx)+T.dot(h_tm1,Wh)+bh)
            y_t = softmax(T.dot(h_t,Wy)+by)
            return [h_t,y_t,hmh]
        h0 = T.zeros((n_hidden,),dtype=theano.config.floatX)
        ([h,y,hmh],_) = theano.scan(fn=step, 
                                sequences=x.dimshuffle([1,0,2]),
                                outputs_info=[T.alloc(h0,x.shape[0],n_hidden),
                                              None,None],
                                non_sequences=[self.Wx,self.Wh,self.Wy,
                                               self.bh,self.by],
                                strict=True)
        self.output = y[-1]
        self.pred = T.argmax(self.output,axis=-1)
        self.idem = T.mean(hmh)
        self.norm = 0.5*T.mean(T.abs_(T.ones(n_hidden)-T.sum(T.sqr(self.Wh),axis=0)))+0.5*T.mean(T.abs_(T.ones(n_hidden)-T.sum(T.sqr(self.Wh),axis=1)))

    def crossentropy(self,y):
        return T.mean(categorical_crossentropy(self.output,y))
    
    def errors(self,y):
        return T.mean(T.neq(self.pred,y))


class rnn_ortho2(object):
    def __init__(self,x,n_in,n_hidden,n_out,bptt_limit):
        """
        Initialize a basic single-layer RNN w/ soft orthogonality penalty
        
        x:    symbolic input tensor
        n_in:    input dimensionality
        n_hidden:    # of hidden units
        hidden_activation:    non-linearity at hidden units (e.g. relu)
        n_out:    # of output units
        output_activation:    non-linearity at output units (e.g. softmax)
        bptt_limit:    # of steps for backprop through time
        """
        self.Wx = _uniform_weight(n_in,n_hidden)
        self.Wh = _ortho_weight(n_hidden)
        self.Wy = _uniform_weight(n_hidden,n_out)
        self.bh = _zero_bias(n_hidden)
        self.by = _zero_bias(n_out)
        self.params = [self.Wx,self.Wh,self.Wy,self.bh,self.by]
        self.W = [self.Wx,self.Wy] # don't incl recurrent weights in decay
        self.L1 = numpy.sum([abs(w).sum() for w in self.W])
        self.L2 = numpy.sum([(w**2).sum() for w in self.W])
        
        def step(x_t,h_tm1,Wx,Wh,Wy,bh,by):
            hWh = T.dot(h_tm1,Wh)
            hmh = T.mean(T.sqr(hWh-T.dot(hWh,Wh)))
            h_t = relu(T.dot(x_t,Wx)+T.dot(h_tm1,Wh)+bh)
            y_t = softmax(T.dot(h_t,Wy)+by)
            return [h_t,y_t,hmh]
        h0 = T.zeros((n_hidden,),dtype=theano.config.floatX)
        ([h,y,hmh],_) = theano.scan(fn=step, 
                                sequences=x.dimshuffle([1,0,2]),
                                outputs_info=[T.alloc(h0,x.shape[0],n_hidden),
                                              None,None],
                                non_sequences=[self.Wx,self.Wh,self.Wy,
                                               self.bh,self.by],
                                strict=True)
        self.output = y[-1]
        self.pred = T.argmax(self.output,axis=-1)
        self.idem = T.mean(hmh)
        self.norm = 0.5*T.mean(T.abs_(T.ones(n_hidden)-T.sum(T.sqr(self.Wh),axis=0)))+0.5*T.mean(T.abs_(T.ones(n_hidden)-T.sum(T.sqr(self.Wh),axis=1)))

    def crossentropy(self,y):
        return T.mean(categorical_crossentropy(self.output,y))
    
    def errors(self,y):
        return T.mean(T.neq(self.pred,y))


class rnn_ortho3(object):
    def __init__(self,x,n_in,n_hidden,n_out,bptt_limit):
        """
        Initialize a basic single-layer RNN
        
        x:    symbolic input tensor
        n_in:    input dimensionality
        n_hidden:    # of hidden units
        hidden_activation:    non-linearity at hidden units (e.g. relu)
        n_out:    # of output units
        output_activation:    non-linearity at output units (e.g. softmax)
        bptt_limit:    # of steps for backprop through time
        """
        self.Wx = _uniform_weight(n_in,n_hidden)
        self.Wh = _ortho_weight(n_hidden)
        self.Wy = _uniform_weight(n_hidden,n_out)
        self.bh = _zero_bias(n_hidden)
        self.by = _zero_bias(n_out)
        self.params = [self.Wx,self.Wh,self.Wy,self.bh,self.by]
        self.W = [self.Wx,self.Wy] # don't incl recurrent weights in decay
        self.L1 = numpy.sum([abs(w).sum() for w in self.W])
        self.L2 = numpy.sum([(w**2).sum() for w in self.W])
        
        def step(x_t,h_tm1,Wx,Wh,Wy,bh,by):
            h_t = relu(T.dot(x_t,Wx)+T.dot(h_tm1,Wh)+bh)
            y_t = softmax(T.dot(h_t,Wy)+by)
            return [h_t,y_t]
        h0 = T.zeros((n_hidden,),dtype=theano.config.floatX)
        ([h,y],_) = theano.scan(fn=step, 
                                sequences=x.dimshuffle([1,0,2]),
                                outputs_info=[T.alloc(h0,x.shape[0],n_hidden),
                                              None],
                                non_sequences=[self.Wx,self.Wh,self.Wy,
                                               self.bh,self.by],
                                strict=True)
        self.output = y[-1]
        self.pred = T.argmax(self.output,axis=-1)
        self.ortho = T.sum(T.sqr(T.dot(self.Wh,self.Wh.T)-T.identity_like(self.Wh)))

    def crossentropy(self,y):
        return T.mean(categorical_crossentropy(self.output,y))
    
    def errors(self,y):
        return T.mean(T.neq(self.pred,y))