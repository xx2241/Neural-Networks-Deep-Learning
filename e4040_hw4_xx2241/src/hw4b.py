"""
Source Code for Homework 4.b of ECBM E4040, Fall 2016, Columbia University

"""

import os
import copy
import timeit
import inspect
import sys
import numpy
from collections import OrderedDict
import random
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample

from hw4_utils import contextwin, shared_dataset, load_data, shuffle, conlleval, check_dir
from hw4_nn import myMLP, train_nn, LogisticRegression

def gen_parity_pair(nbit, num):
    """
    Generate binary sequences and their parity bits

    :type nbit: int
    :param nbit: length of binary sequence

    :type num: int
    :param num: number of sequences

    """
    X = numpy.random.randint(2, size=(num, nbit))
    Y = numpy.mod(numpy.sum(X, axis=1), 2)
    
    return X, Y

def gen_rnn_parity_pair(nbit, num):
    X = numpy.random.randint(2, size=(num, nbit))
    Y = numpy.array(X)
    for i in range(1,nbit):
        Y[:,i] += Y[:, i-1]

    Y = numpy.mod(Y, 2)
    return X, Y

#TODO: implement RNN class to learn parity function
class RNN(object):
    """ Elman Neural Net Model Class
    """
    def __init__(self, nh, nc, cs):
        """Initialize the parameters for the RNNSLU


        """
        # parameters of the model
        
        self.wx = theano.shared(name='wx',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (cs, nh))
                                .astype(theano.config.floatX))
        self.wh = theano.shared(name='wh',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.w = theano.shared(name='w',
                               value=0.2 * numpy.random.uniform(-1.0, 1.0,
                               (nh, nc))
                               .astype(theano.config.floatX))
        self.bh = theano.shared(name='bh',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.b = theano.shared(name='b',
                               value=numpy.zeros(nc,
                               dtype=theano.config.floatX))
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))

        # bundle
        self.params = [self.wx, self.wh, self.w, self.bh, self.b, self.h0]

        # as many columns as context window size
        # as many lines as words in the sequence
        x = T.matrix()
        y_sequence = T.ivector('y_sequence')  # labels

        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.wx) + T.dot(h_tm1, self.wh) + self.bh)
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence,
                                sequences=x,
                                outputs_info=[self.h0, None],
                                n_steps=x.shape[0])

        p_y_given_x_sequence = s[:, 0, :]
        y_pred = T.argmax(p_y_given_x_sequence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')

        sequence_nll = -T.mean(T.log(p_y_given_x_sequence)
                               [T.arange(x.shape[0]), y_sequence])

        sequence_gradients = T.grad(sequence_nll, self.params)

        sequence_updates = OrderedDict((p, p - lr*g)
                                       for p, g in
                                       zip(self.params, sequence_gradients))

        # theano functions to compile
        self.classify = theano.function(inputs=[x], outputs=y_pred, allow_input_downcast=True)
        self.sequence_train = theano.function(inputs=[x, y_sequence, lr],
                                              outputs=sequence_nll,
                                              updates=sequence_updates,
                                              allow_input_downcast=True)
        self.error = T.mean(T.sqr(y_pred-y_sequence))

    def train(self, x, y, window_size, learning_rate):
        cwords = contextwin(x, window_size)
        words = list(map(lambda x: numpy.asarray(x).astype('int32'), cwords))
        labels = y

        self.sequence_train(words, labels, learning_rate)

        
    

#TODO: implement LSTM class to learn parity function
class LSTM(object):
    
    def __init__(self, nh, nc, cs):
        """Initialize the parameters for the RNNSLU


        """
        # parameters of the model
        self.wf = theano.shared(name='wf',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (cs, nh))
                                .astype(theano.config.floatX))
        self.wi = theano.shared(name='wi',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (cs, nh))
                                .astype(theano.config.floatX))
        self.wo = theano.shared(name='wo',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (cs, nh))
                                .astype(theano.config.floatX))
        self.wc = theano.shared(name='wc',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (cs, nh))
                                .astype(theano.config.floatX))
        self.uf = theano.shared(name='uf',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.ui = theano.shared(name='ui',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.uo = theano.shared(name='uo',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.uc = theano.shared(name='uc',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))

        self.bf = theano.shared(name='bf',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bi = theano.shared(name='bi',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bo = theano.shared(name='bo',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bc = theano.shared(name='bc',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))

        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.c0 = theano.shared(name='c0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.w = theano.shared(name='w',
                               value=0.2 * numpy.random.uniform(-1.0, 1.0,
                               (nh, nc))
                               .astype(theano.config.floatX))
        self.b = theano.shared(name='b',
                               value=numpy.zeros(nc,
                               dtype=theano.config.floatX))
        self.params = [self.wf, self.wi, self.wo, self.wc, self.uf, self.ui, self.uo, self.uc, self.bf, self.bi, self.bo, self.bc, self.w,self.b]

        # as many columns as context window size
        # as many lines as words in the sequence
        x = T.matrix()
        y_sequence = T.ivector('y_sequence')  # labels

        def recurrence(x_t, c_tm1, h_tm1):
            f_t = T.nnet.sigmoid(T.dot(x_t, self.wf) + T.dot(h_tm1, self.uf) + self.bf)
            i_t = T.nnet.sigmoid(T.dot(x_t, self.wi) + T.dot(h_tm1, self.ui) + self.bi)
            o_t = T.nnet.sigmoid(T.dot(x_t, self.wo) + T.dot(h_tm1, self.uo) + self.bo)
            c_t = f_t * c_tm1 + i_t * T.tanh(T.dot(x_t, self.wc) + T.dot(h_tm1, self.uc) + self.bc) 
            h_t = o_t * T.tanh(c_t)
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return [c_t, h_t, s_t]

        [c ,h, s], _ = theano.scan(fn=recurrence,
                                sequences=x,
                                outputs_info=[self.c0, self.h0, None],
                                n_steps=x.shape[0])

        p_y_given_x_sequence = s[:, 0, :]
        y_pred = T.argmax(p_y_given_x_sequence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')

        sequence_nll = -T.mean(T.log(p_y_given_x_sequence)
                               [T.arange(x.shape[0]), y_sequence])

        sequence_gradients = T.grad(sequence_nll, self.params)

        sequence_updates = OrderedDict((p, p - lr*g)
                                       for p, g in
                                       zip(self.params, sequence_gradients))

        # theano functions to compile
        self.classify = theano.function(inputs=[x], outputs=y_pred, allow_input_downcast=True)
        self.sequence_train = theano.function(inputs=[x, y_sequence, lr],
                                              outputs=sequence_nll,
                                              updates=sequence_updates,
                                              allow_input_downcast=True)
        self.error = T.mean(T.sqr(y_pred-y_sequence))

    def train(self, x, y, window_size, learning_rate):
        cwords = contextwin(x, window_size)
        words = list(map(lambda x: numpy.asarray(x).astype('int32'), cwords))
        labels = y

        self.sequence_train(words, labels, learning_rate)
        


        
        







#TODO: build and train a MLP to learn parity function
def test_mlp_parity(learning_rate=0.01, n_bit=8,n_hiddenLayers=2,L1_reg=0.00, L2_reg=0.0001, n_epochs=100, batch_size=100, n_hidden=500, patience=10000, verbose=True):
    
    # generate datasets
    train_set = gen_parity_pair(n_bit, 1000)
    valid_set = gen_parity_pair(n_bit, 500)
    test_set  = gen_parity_pair(n_bit, 100)

    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)
   
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size
    
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    rng = numpy.random.RandomState(1234)

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    classifier = myMLP(rng=rng, input=x, n_in=n_bit, n_hidden=n_hidden, n_out=2, n_hiddenLayers=n_hiddenLayers)

    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    
    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    print('...training')
    
    train_nn(train_model, validate_model, test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs, patience, verbose)

    
#TODO: build and train a RNN to learn parity function
def test_rnn_parity(**kwargs):
    

    param = {
        'nbits':8,
        'lr': 0.1,
        'verbose': True,
        'decay': True,
        'win': 7,
        'nhidden': 5,
        'nepochs': 60
    }

    param_diff = set(kwargs.keys()) - set(param.keys())
    if param_diff:
        raise KeyError("invalid arguments:" + str(tuple(param_diff)))
    param.update(kwargs)

    if param['verbose']:
        for k,v in param.items():
            print("%s: %s" % (k,v))

  
    
    numpy.random.seed(100)  # Gaurantees consistency across runs
    train_x, train_y = gen_rnn_parity_pair(param['nbits'], 1000)
    valid_x, valid_y = gen_rnn_parity_pair(param['nbits'], 500)
    test_x, test_y  = gen_rnn_parity_pair(param['nbits'], 100)
    

    nclasses=2
    nsequences = len(train_x)
    random.seed(100)

    print('... building the model')
    rnn = RNN(
        nh=param['nhidden'],
        nc=nclasses,
        cs=param['win']
    )

    # train with early stopping on validation set
    print('... training')
    best_error = 100
    param['clr'] = param['lr']
    extract_col = train_x.shape[1]-1

    for e in range(param['nepochs']):

        param['ce'] = e
        tic = timeit.default_timer()

        for i, (x, y) in enumerate(zip(train_x, train_y)):
            rnn.train(numpy.asarray(x), numpy.asarray(y), param['win'], param['clr'])
            sys.stdout.flush()

        # evaluation and prediction

        pred_valid = numpy.asarray([rnn.classify(numpy.asarray( contextwin(x, param['win'])).astype('int32')) for x in valid_x ])
        pred_test = numpy.asarray([rnn.classify(numpy.asarray( contextwin(x, param['win'])).astype('int32')) for x in test_x ])



        valid_error = numpy.mean((valid_y[:, extract_col] - pred_valid[:, extract_col])**2)
        test_error = numpy.mean((test_y[:, extract_col] - pred_test[:, extract_col])**2)
 

       

        if valid_error < best_error:

            #best_rnn = copy.deepcopy(rnn)
            best_error = valid_error


            if param['verbose']:
                print('NEW BEST: epoch', e,
                      'validation error', valid_error,
                      'best test test error', test_error)

            param['valid_error'], param['test_error'] = valid_error, test_error
            param['be'] = e


        # learning rate decay if no improvement in 10 epochs
        if param['decay'] and abs(param['be']-param['ce']) >= 10:
            param['clr'] *= 0.5

        if param['clr'] < 1e-5:
            break

    print('BEST RESULT: epoch', param['be'],
           'validation error', param['valid_error'],
           'best test error', param['test_error'])
   

    
    
#TODO: build and train a LSTM to learn parity function
def test_lstm_parity(**kwargs):
    

    param = {
        'nbits':8,
        'lr': 0.1,
        'verbose': True,
        'decay': True,
        'win': 7,
        'nhidden': 1,
        'nepochs': 20
    }

    param_diff = set(kwargs.keys()) - set(param.keys())
    if param_diff:
        raise KeyError("invalid arguments:" + str(tuple(param_diff)))
    param.update(kwargs)

    if param['verbose']:
        for k,v in param.items():
            print("%s: %s" % (k,v))

  
    
    numpy.random.seed(100)  # Gaurantees consistency across runs
    train_x, train_y = gen_rnn_parity_pair(param['nbits'], 1000)
    valid_x, valid_y = gen_rnn_parity_pair(param['nbits'], 500)
    test_x, test_y  = gen_rnn_parity_pair(param['nbits'], 100)
    

    nclasses=2
    nsequences = len(train_x)
    random.seed(100)

    print('... building the model')
    rnn = LSTM(
        nh=param['nhidden'],
        nc=nclasses,
        cs=param['win']
    )

    # train with early stopping on validation set
    print('... training')
    best_error = 100
    param['clr'] = param['lr']
    extract_col = train_x.shape[1]-1

    for e in range(param['nepochs']):

        param['ce'] = e
        tic = timeit.default_timer()

        for i, (x, y) in enumerate(zip(train_x, train_y)):
            rnn.train(numpy.asarray(x), numpy.asarray(y), param['win'], param['clr'])
            sys.stdout.flush()

        # evaluation and prediction

        pred_valid = numpy.asarray([rnn.classify(numpy.asarray( contextwin(x, param['win'])).astype('int32')) for x in valid_x ])
        pred_test = numpy.asarray([rnn.classify(numpy.asarray( contextwin(x, param['win'])).astype('int32')) for x in test_x ])



        valid_error = numpy.mean((valid_y[:, extract_col] - pred_valid[:, extract_col])**2)
        test_error = numpy.mean((test_y[:, extract_col] - pred_test[:, extract_col])**2)
 

       

        if valid_error < best_error:

            #best_rnn = copy.deepcopy(rnn)
            best_error = valid_error


            if param['verbose']:
                print('NEW BEST: epoch', e,
                      'validation error', valid_error,
                      'best test test error', test_error)

            param['valid_error'], param['test_error'] = valid_error, test_error
            param['be'] = e


        # learning rate decay if no improvement in 10 epochs
        if param['decay'] and abs(param['be']-param['ce']) >= 10:
            param['clr'] *= 0.5

        if param['clr'] < 1e-5:
            break

    print('BEST RESULT: epoch', param['be'],
           'validation error', param['valid_error'],
           'best test error', param['test_error'])
   

  

    
if __name__ == '__main__':
    '''
    test_lstm_parity(**{
        'nbits':8,
        'lr': 0.1,
        'verbose': True,
        'decay': True,
        'win': 1,
        'nhidden': 1,
        'nepochs': 20
    })
    
    test_lstm_parity(**{
        'nbits':12,
        'lr': 0.1,
        'verbose': True,
        'decay': True,
        'win': 1,
        'nhidden': 1,
        'nepochs': 20
    })
    '''
    test_rnn_parity(**{
        'nbits':8,
        'lr': 0.2,
        'verbose': True,
        'decay': True,
        'win': 1,
        'nhidden': 5,
        'nepochs': 20
    })
    '''
    test_rnn_parity(**{
        'nbits':12,
        'lr': 1,
        'verbose': True,
        'decay': True,
        'win': 1,
        'nhidden': 10,
        'nepochs': 30
    })
    '''

