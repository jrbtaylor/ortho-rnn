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

rng0 = numpy.random.RandomState(1)


def _uniform_weight(n1,n2,rng=rng0):
    limit = numpy.sqrt(6./(n1+n2))
    return theano.shared((rng.uniform(low=-limit,
                                      high=limit,
                                      size=(n1,n2))
                         ).astype(theano.config.floatX),
                         borrow=True)


def _ortho_weight(n,rng=rng0):
    W = rng.randn(n, n)
    u, s, v = numpy.linalg.svd(W)
    return theano.shared(u.astype(theano.config.floatX),
                         borrow=True)


def _zero_bias(n):
    return theano.shared(numpy.zeros((n,),dtype=theano.config.floatX),
                         borrow=True)


class rnn(object):
    def __init__(self,x,n_in,n_hidden,n_out,rng=rng0):
        """
        Initialize a basic single-layer RNN
        
        x:    symbolic input tensor
        n_in:    input dimensionality
        n_hidden:    # of hidden units
        hidden_activation:    non-linearity at hidden units (e.g. relu)
        n_out:    # of output units
        output_activation:    non-linearity at output units (e.g. softmax)
        """
        self.Wx = _uniform_weight(n_in,n_hidden,rng)
        self.Wh = _ortho_weight(n_hidden,rng)
        self.Wy = _uniform_weight(n_hidden,n_out,rng)
        self.bh = _zero_bias(n_hidden)
        self.by = _zero_bias(n_out)
        self.params = [self.Wx,self.Wh,self.Wy,self.bh,self.by]
        self.W = [self.Wx,self.Wy] # don't incl recurrent weights in decay
        self.L1 = numpy.sum([abs(w).sum() for w in self.W])
        self.L2 = numpy.sum([(w**2).sum() for w in self.W])
        
        def step(x_t,h_tm1,Wx,Wh,Wy,bh,by):
            h_t = sigmoid(T.dot(x_t,Wx)+T.dot(h_tm1,Wh)+bh)
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
        weight_norms = [T.sqrt(T.sum(T.sqr(w))) for w in [self.Wx,self.Wh,self.Wy]]
        orthogonality = T.sum(T.sqr(T.dot(self.Wh,self.Wh.T)-T.identity_like(self.Wh)))
        self.monitors = T.as_tensor_variable(weight_norms+[orthogonality])

    def crossentropy(self,y):
        return T.mean(categorical_crossentropy(self.output,y))
    
    def errors(self,y):
        return T.mean(T.neq(self.pred,y))


class rnn_ortho(rnn):
    def __init__(self,x,n_in,n_hidden,n_out,rng=rng0):
        super(rnn_ortho,self).__init__(x,n_in,n_hidden,n_out,rng)
        self.ortho = T.sum(T.sqr(T.dot(self.Wh,self.Wh.T)- \
                                 T.identity_like(self.Wh)))


class lstm(object):
    def __init__(self,x,n_in,n_hidden,n_out,rng=rng0):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        # initialize weights
        self.Wxi = _uniform_weight(n_in,n_hidden)
        self.Wsi = _ortho_weight(n_hidden,rng)
        self.Wxf = _uniform_weight(n_in,n_hidden)
        self.Wsf = _ortho_weight(n_hidden,rng)
        self.Wxo = _uniform_weight(n_in,n_hidden)
        self.Wso = _ortho_weight(n_hidden,rng)
        self.Wxg = _uniform_weight(n_in,n_hidden)
        self.Wsg = _ortho_weight(n_hidden,rng)
        self.Wsy = _uniform_weight(n_hidden,n_out)
        self.bi = _zero_bias(n_hidden)
        self.bf = _zero_bias(n_hidden)
        self.bo = _zero_bias(n_hidden)
        self.bg = _zero_bias(n_hidden)
        self.params = [self.Wxi,self.Wsi,self.Wxf,self.Wsf,self.Wxo,self.Wso,self.Wxg,self.Wsg,self.Wsy,self.bi,self.bf,self.bo,self.bg]
        self.W = [self.Wxi,self.Wsi,self.Wxf,self.Wsf,self.Wxo,self.Wso,self.Wxg,self.Wsg,self.Wsy]
        self.L1 = numpy.sum([abs(w).sum() for w in self.W])
        self.L2 = numpy.sum([(w**2).sum() for w in self.W])
        # forward function
        def forward(x_t,c_tm1,s_tm1,Wxi,Wsi,Wxf,Wsf,Wxo,Wso,Wxg,Wsg,Wsy,bi,bf,bo,bg):
            i = sigmoid(T.dot(x_t,Wxi)+T.dot(s_tm1,Wsi)+bi)
            f = sigmoid(T.dot(x_t,Wxf)+T.dot(s_tm1,Wsf)+bf)
            o = sigmoid(T.dot(x_t,Wxo)+T.dot(s_tm1,Wso)+bo)
            g = tanh(T.dot(x_t,Wxg)+T.dot(s_tm1,Wsg)+bg)
            c = c_tm1*f+g*i
            s = tanh(c)*o
            y = softmax(T.dot(s,Wsy))
            return [c,s,y]
        c0 = T.alloc(T.zeros((self.n_hidden,),dtype=theano.config.floatX),x.shape[0],self.n_hidden)
        s0 = T.alloc(T.zeros((self.n_hidden,),dtype=theano.config.floatX),x.shape[0],self.n_hidden)
        ([c,s,y],updates) = theano.scan(fn=forward,
                                      sequences=x.dimshuffle([1,0,2]),
                                      outputs_info=[c0,s0,None],
                                      non_sequences=[self.Wxi,self.Wsi,self.Wxf,self.Wsf,self.Wxo,self.Wso,self.Wxg, \
                                                     self.Wsg,self.Wsy,self.bi,self.bf,self.bo,self.bg],
                                      strict=True)
        self.output = y[-1]
        self.pred = T.argmax(self.output,axis=1)
        weight_norms = [T.sqrt(T.sum(T.sqr(w))) for w in [self.Wxg,self.Wsg,self.Wsy]]
        orthogonality = T.sum(T.sqr(T.dot(self.Wsg,self.Wsg.T)-T.identity_like(self.Wsg)))
        self.monitors = T.as_tensor_variable(weight_norms+[orthogonality])
    
    # ----- Classification -----
    def crossentropy(self,y):
        return T.mean(categorical_crossentropy(self.output,y))
    
    def errors(self,y):
        return T.mean(T.neq(self.pred,y))
        

class lstm_ortho(lstm):
    def __init__(self,x,n_in,n_hidden,n_out,rng=rng0):
        super(lstm_ortho,self).__init__(x,n_in,n_hidden,n_out,rng)
        self.ortho = T.sum(T.sqr(T.dot(self.Wsg,self.Wsg.T)- \
                           T.identity_like(self.Wsg)))








