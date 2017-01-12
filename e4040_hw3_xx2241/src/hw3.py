"""
Source Code for Homework 3 of ECBM E4040, Fall 2016, Columbia University

Instructor: Prof. Zoran Kostic

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""
import numpy
import sys
import os
import timeit
import scipy.ndimage
import theano
import theano.tensor as T
import numpy as np
import random
from theano.tensor.signal import downsample
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

from hw3_utils import shared_dataset, load_data
from hw3_nn import LogisticRegression, HiddenLayer, LeNetConvPoolLayer, train_nn

class ConvLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, border_mode='half',poolsize=(1, 1)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input


        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode=border_mode
        )
        self.output = T.nnet.relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

#Problem 1
#Implement the convolutional neural network architecture depicted in HW3 problem 1
#Reference code can be found in http://deeplearning.net/tutorial/code/convolutional_mlp.py

def test_lenet(learning_rate=0.1, n_epochs=200, nkerns=[32, 64], batch_size=500):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    


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

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 3, 3),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 15, 15),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 6 * 6,
        n_out=4096,
        activation=T.tanh
    )
    layer3 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=4096,
        n_out=512,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(input=layer3.output, n_in=512, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    train_nn(train_model, validate_model, test_model,n_train_batches, n_valid_batches, n_test_batches, n_epochs,verbose=True)


#Problem 2.1
#Write a function to add translations
def translate_image(input_image,rang):
    rng0 = np.random.randint(rang)
    rng1 = np.random.randint(rang)
    rng2 = np.random.randint(4)
    rng3 = np.random.randint(4)
    rng5 = np.random.randint(4)
    shift_image=np.copy(input_image)
    if rng5<2:
        return shift_image
    if rng2<2:
        shift_image[:,0:32*rng0]=0
        shift_image[:,1024:1024+32*rng0]=0
        shift_image[:,2048:2048+32*rng0]=0
    else:
        shift_image[:,1024-32*rng0:1024]=0
        shift_image[:,2048-32*rng0:2048]=0
        shift_image[:,3072-32*rng0:3072]=0
    if rng3<2:
        for i in range(rng1):
            shift_image[:,i:1024:32]=0
            shift_image[:,1024+i:2048:32]=0
            shift_image[:,2048+i:3072:32]=0
    else:
        for i in range(rng1):
            shift_image[:,31-i:1024:32]=0
            shift_image[:,1024+31-i:2048:32]=0
            shift_image[:,2048+31-i:3072:32]=0
    return shift_image


#Implement a convolutional neural network with the translation method for augmentation
def test_lenet_translation(learning_rate=0.1, n_epochs=200, nkerns=[32, 64], batch_size=500):
    
    rng = numpy.random.RandomState(23455)
   
    datasets = load_data(ds_rate = None, theano_shared = False)

    train_set_x, train_set_y = datasets[0]
    train_set_y = train_set_y.astype(int)
    
    valid_set_x, valid_set_y = shared_dataset(datasets[1])
    test_set_x, test_set_y = shared_dataset(datasets[2])

    

    
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.shape[0]
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

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 3, 3),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 15, 15),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 6 * 6,
        n_out=4096,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    #layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)
    layer3 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=4096,
        n_out=512,
        activation=T.tanh
    )
    
    layer4 = LogisticRegression(input=layer3.output, n_in=512, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [x,y],
        cost,
        updates=updates,
        allow_input_downcast=True
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.85  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        augment_set_x = translate_image(train_set_x,2)
        for minibatch_index in range(n_train_batches):
            


            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
                
            
            cost_ij = train_model(augment_set_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size],train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size])

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))
    



#Problem 2.2
#Write a function to add roatations
def rotate_image(input_image,scale):
    #how to rotate matrix all at one
    #how to add random variable
    angle = random.randrange(-scale,scale)
    output_image = np.copy(input_image)
    output_image = output_image.reshape(output_image.shape[0],3,32,32)
    output_image = scipy.ndimage.rotate(output_image, axes=(2,3),angle=angle,reshape=False)
    output_image = output_image.reshape(output_image.shape[0],3072)
    return output_image
#Implement a convolutional neural network with the rotation method for augmentation
def test_lenet_rotation(learning_rate=0.1, n_epochs=200, nkerns=[32, 64], batch_size=500):
    
    rng = numpy.random.RandomState(23455)
   
    datasets = load_data(ds_rate = None, theano_shared = False)

    train_set_x, train_set_y = datasets[0]
    train_set_y = train_set_y.astype(int)
    
    valid_set_x, valid_set_y = shared_dataset(datasets[1])
    test_set_x, test_set_y = shared_dataset(datasets[2])

    

    
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.shape[0]
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

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 3, 3),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 15, 15),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 6 * 6,
        n_out=4096,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    #layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)
    layer3 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=4096,
        n_out=512,
        activation=T.tanh
    )
    
    layer4 = LogisticRegression(input=layer3.output, n_in=512, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [x,y],
        cost,
        updates=updates,
        allow_input_downcast=True
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.85  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        augment_set_x = rotate_image(train_set_x,10)
        for minibatch_index in range(n_train_batches):
            


            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
                
            
            cost_ij = train_model(augment_set_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size],train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size])

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))

#Problem 2.3
#Write a function to flip images
def flip_image(input_image):
    rng = np.random.randint(6)
    output_image = np.copy(input_image)
    if rng<3:
        output_image = output_image.reshape(output_image.shape[0],3,32,32)
        for i in range(output_image.shape[0]):
            img=output_image[i].transpose(1,2,0)
            img=np.fliplr(img)
            output_image[i]=img.transpose(2,0,1)
        output_image = output_image.reshape(output_image.shape[0],3072)
        return output_image
    else:
        return output_image


                      
#Implement a convolutional neural network with the flip method for augmentation
def test_lenet_flip(learning_rate=0.1, n_epochs=200, nkerns=[32, 64], batch_size=500):
    
    rng = numpy.random.RandomState(23455)
   
    datasets = load_data(ds_rate = None, theano_shared = False)

    train_set_x, train_set_y = datasets[0]
    train_set_y = train_set_y.astype(int)
    
    valid_set_x, valid_set_y = shared_dataset(datasets[1])
    test_set_x, test_set_y = shared_dataset(datasets[2])

    

    
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.shape[0]
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

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 3, 3),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 15, 15),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 6 * 6,
        n_out=4096,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    #layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)
    layer3 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=4096,
        n_out=512,
        activation=T.tanh
    )
    
    layer4 = LogisticRegression(input=layer3.output, n_in=512, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [x,y],
        cost,
        updates=updates,
        allow_input_downcast=True
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    # early-stopping parameters
    patience = 4000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.85  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        augment_set_x = flip_image(train_set_x)
        for minibatch_index in range(n_train_batches):
            


            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
            
            cost_ij = train_model(augment_set_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size],train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size])

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))       
#Problem 2.4
#Write a function to add noise, it should at least provide Gaussian-distributed and uniform-distributed noise with zero mean
def noise_injection(input_image, scale):
    rng = np.random.randint(3)
    output_image = np.copy(input_image)
    if rng==0:
        output_image = output_image + np.random.normal(loc=0, scale = scale, size = output_image.shape)
    if rng==1:
        output_image = output_image + np.random.uniform(low=-1*scale, high = scale, size = output_image.shape)
    return output_image
        
#Implement a convolutional neural network with the augmentation of injecting noise into input
def test_lenet_inject_noise_input(learning_rate=0.1, n_epochs=200, nkerns=[32, 64], batch_size=500):
    
    rng = numpy.random.RandomState(23455)
   
    datasets = load_data(ds_rate = None, theano_shared = False)

    train_set_x, train_set_y = datasets[0]
    train_set_y = train_set_y.astype(int)
    
    valid_set_x, valid_set_y = shared_dataset(datasets[1])
    test_set_x, test_set_y = shared_dataset(datasets[2])

    

    
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.shape[0]
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

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 3, 3),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 15, 15),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 6 * 6,
        n_out=4096,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    #layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)
    layer3 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=4096,
        n_out=512,
        activation=T.tanh
    )
    
    layer4 = LogisticRegression(input=layer3.output, n_in=512, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [x,y],
        cost,
        updates=updates,
        allow_input_downcast=True
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.85  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        augment_set_x = noise_injection(train_set_x,0.01)
        for minibatch_index in range(n_train_batches):
            


            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
                
            
            cost_ij = train_model(augment_set_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size],train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size])

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))
    
#Problem 3 
#Implement a convolutional neural network to achieve at least 80% testing accuracy on CIFAR-dataset
def drop(input, p=0.5): 
    """
    :type input: numpy.array
    :param input: layer or weight matrix on which dropout is applied
    
    :type p: float or double between 0. and 1. 
    :param p: p probability of NOT dropping out a unit, therefore (1.-p) is the drop rate.
    
    """            
    rng = numpy.random.RandomState(1234)
    srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
    return input * mask

class DropoutHiddenLayer(object):
    def __init__(self, rng, is_train, input, n_in, n_out, W=None, b=None,
                 activation=T.nnet.relu, p=0.5):
        """
        Hidden unit activation is given by: activation(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        
        :type is_train: theano.iscalar   
        :param is_train: indicator pseudo-boolean (int) for switching between training and prediction

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
                           
        :type p: float or double
        :param p: probability of NOT dropping out a unit   
        """
        self.input = input

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        
        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        
        output = activation(lin_output)
        
        # multiply output and drop -> in an approximation the scaling effects cancel out 
        train_output = drop(output,p)
        
        #is_train is a pseudo boolean theano variable for switching between training and prediction 
        self.output = T.switch(T.neq(is_train, 0), train_output, p*output)
        
        # parameters of the model
        self.params = [self.W, self.b]
        
def adam(loss, all_params, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8,
         gamma=1-1e-8):
    """
    ADAM update rules
    Default values are taken from [Kingma2014]
    References:
    [Kingma2014] Kingma, Diederik, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    http://arxiv.org/pdf/1412.6980v4.pdf
    """
    updates = []
    all_grads = theano.grad(loss, all_params)
    alpha = learning_rate
    t = theano.shared(np.float32(1))
    b1_t = b1*gamma**(t-1)   #(Decay the first moment running average coefficient)

    for theta_previous, g in zip(all_params, all_grads):
        m_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))
        v_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))

        m = b1_t*m_previous + (1 - b1_t)*g                             # (Update biased first moment estimate)
        v = b2*v_previous + (1 - b2)*g**2                              # (Update biased second raw moment estimate)
        m_hat = m / (1-b1**t)                                          # (Compute bias-corrected first moment estimate)
        v_hat = v / (1-b2**t)                                          # (Compute bias-corrected second raw moment estimate)
        theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e) #(Update parameters)

        updates.append((m_previous, m))
        updates.append((v_previous, v))
        updates.append((theta_previous, theta) )
    updates.append((t, t + 1.))
    return updates

def MY_lenet(n_epochs=200, nkerns=[32, 64, 128], batch_size=500, activation=T.nnet.relu):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    datasets = load_data(theano_shared = False)

    train_set_x, train_set_y = datasets[0]
    
    
    valid_set_x, valid_set_y = shared_dataset(datasets[1])
    test_set_x, test_set_y = shared_dataset(datasets[2])

    


    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.shape[0]
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
    training_enabled = T.iscalar('training_enabled') # pseudo boolean for switching between training and prediction
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layerConv1_input = x.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layerConv1 = ConvLayer(
        rng,
        input=layerConv1_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 3, 3),
        border_mode='half',
        poolsize=(1, 1)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layerConv2 = ConvLayer(
        rng,
        input=layerConv1.output,
        image_shape=(batch_size, nkerns[0], 32, 32),
        filter_shape=(nkerns[0], nkerns[0], 3, 3),
        border_mode='half',
        poolsize=(2, 2)
    )
    #output size(3,32,32)
    pool_out1 = pool.pool_2d(
        input = layerConv2.output,
        ds = (2,2),
        ignore_border=True
    )
    #output size(nkern[0],16,16)
    layerConv3 = ConvLayer(
        rng,
        input=pool_out1,
        image_shape=(batch_size, nkerns[0], 16, 16),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        border_mode='half',
        poolsize=(1, 1)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layerConv4 = ConvLayer(
        rng,
        input=layerConv3.output,
        image_shape=(batch_size, nkerns[1], 16, 16),
        filter_shape=(nkerns[1], nkerns[1], 3, 3),
        border_mode='half',
        poolsize=(2, 2)
    )
    #output size(nkerns[1],16,16)
    pool_out2 = pool.pool_2d(
        input = layerConv4.output,
        ds = (2,2),
        ignore_border=True
    )
    #output size(nkerns[1],8,8)
    layerConv5 = ConvLayer(
        rng,
        input=pool_out2,
        image_shape=(batch_size, nkerns[1], 8, 8),
        filter_shape=(nkerns[2], nkerns[1], 3, 3),
        border_mode='half',
        poolsize=(1, 1)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layerConv6 = ConvLayer(
        rng,
        input=layerConv5.output,
        image_shape=(batch_size, nkerns[2], 8, 8),
        filter_shape=(nkerns[2], nkerns[2], 3, 3),
        border_mode='half',
        poolsize=(2, 2)
    )
    #output size(nkern[1],8,8)
    pool_out3 = pool.pool_2d(
        input = layerConv6.output,
        ds = (2,2),
        ignore_border=True
    )
    #output size(nkerns[2],4,4)


    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = pool_out3.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = DropoutHiddenLayer(
        rng,
        input=layer2_input,
        is_train = training_enabled,
        n_in=nkerns[2] * 4 * 4,
        n_out=4096,
        activation=activation
    )
    layer3 = DropoutHiddenLayer(
        rng,
        input=layer2.output,
        is_train = training_enabled,
        n_in=4096,
        n_out=512,
        activation=activation
    )

    # classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(input=layer3.output, n_in=512, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            training_enabled: numpy.cast['int32'](0)
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
            training_enabled: numpy.cast['int32'](0)
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layerConv1.params + layerConv2.params + layerConv3.params + layerConv4.params + layerConv5.params + layerConv6.params + layer2.params + layer3.params + layer4.params



    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    
    #using adam!!!!!!!!!
    updates = adam(cost,params)

    train_model = theano.function(
        [x,y,training_enabled],
        cost,
        updates=updates,
        allow_input_downcast=True
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 20000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            


            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
                
            augrng = np.random.randint(3)  
            if augrng==0:
                augment_set_x = translate_image(train_set_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size],2)
            if augrng==1:
                augment_set_x = rotate_image(train_set_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size],4)
            if augrng==2:
                augment_set_x = noise_injection(train_set_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size],0.006)
                
            cost_ij = train_model(augment_set_x,train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size],numpy.cast['int32'](1))

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))

#Problem4
#Implement the convolutional neural network depicted in problem4

def dropout(input_set, p=0.7): 
    """
    :type input: numpy.array
    :param input: layer or weight matrix on which dropout is applied
    
    :type p: float or double between 0. and 1. 
    :param p: p probability of NOT dropping out a unit, therefore (1.-p) is the drop rate.
    
    """
    mask = np.random.rand(*input_set.shape) < p
    return input_set*mask
    
class outputlayerclass(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input):
        self.input = input

    def mse(self, y): 
        return T.mean(T.sqr(self.input-y))

    def errors(self, y):
        return T.mean(T.sqr(self.input-y))
    

    
def MY_CNN(learning_rate=0.1, n_epochs=128, nkerns=[32, 64, 128, 256], batch_size=300):
    
    rng = numpy.random.RandomState(23455)

    datasets = load_data(theano_shared=False)

    train_set_y, train_set_no = datasets[0]    
    valid_set_y, valid_set_no = datasets[1]
    test_set_y, test_set_no = datasets[2]
    not_shared_test_set_y = test_set_y
    

    #use this set to output the final figure

    
    train_set_x = dropout(train_set_y, p=0.7)
    valid_set_x = dropout(valid_set_y, p=0.7)
    test_set_x = dropout(test_set_y, p=0.7)
    not_shared_test_set_x = test_set_x
    
   
    
    
    train_set_x = theano.shared(numpy.asarray(train_set_x,
                                           dtype=theano.config.floatX),
                             borrow=True)
    train_set_y = theano.shared(numpy.asarray(train_set_y,
                                           dtype=theano.config.floatX),
                             borrow=True)
    valid_set_x = theano.shared(numpy.asarray(valid_set_x,
                                           dtype=theano.config.floatX),
                             borrow=True)
    valid_set_y = theano.shared(numpy.asarray(valid_set_y,
                                           dtype=theano.config.floatX),
                             borrow=True)
    test_set_x = theano.shared(numpy.asarray(test_set_x,
                                           dtype=theano.config.floatX),
                             borrow=True)
    test_set_y = theano.shared(numpy.asarray(test_set_y,
                                           dtype=theano.config.floatX),
                             borrow=True)
    


    


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
    y = T.matrix('y')  # the labels are presented as 1D vector of
                        # [int] labels    

    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')
    
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    layer0 = ConvLayer(
        rng,
        input = layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[1], 3, 3, 3),
        poolsize=(1, 1)
    )
    
    layer1 = ConvLayer(
        rng,
        input = layer0.output,
        image_shape=(batch_size, nkerns[1], 32, 32),
        filter_shape=(nkerns[1], nkerns[1], 3, 3),
        poolsize = (2,2)
    )

    
    pool_out1 = pool.pool_2d(
        input = layer1.output,
        ds = (2,2),
        ignore_border=True
    )

    layer2 = ConvLayer(
        rng,
        input = pool_out1,
        image_shape=(batch_size, nkerns[1], 16, 16),
        filter_shape=(nkerns[2], nkerns[1], 3, 3),
        poolsize=(1, 1)
    )
    
    layer3 = ConvLayer(
        rng,
        input = layer2.output,
        image_shape=(batch_size, nkerns[2], 16, 16),
        filter_shape=(nkerns[2], nkerns[2], 3, 3),
        poolsize=(2, 2)
    )
    
    pool_out2 = pool.pool_2d(
        input = layer3.output,
        ds = (2,2),
        ignore_border=True
    )
    
    layer4 = ConvLayer(
        rng,
        input = pool_out2,
        image_shape=(batch_size, nkerns[2], 8, 8),
        filter_shape=(nkerns[3], nkerns[2], 3, 3),
        poolsize=(0.5, 0.5)
    ) 
    
    #can poolsize be set into 0.5(float)?
    upsampling1 = T.extra_ops.repeat(layer4.output, 2, axis=2)
    upsampling1 = T.extra_ops.repeat(upsampling1, 2, axis=3)
    
    layer5 = ConvLayer(
        rng,
        input = upsampling1,
        image_shape=(batch_size, nkerns[3], 16, 16),
        filter_shape=(nkerns[2], nkerns[3], 3, 3),
        poolsize=(1, 1)
    ) 
    layer6 = ConvLayer(
        rng,
        input = layer5.output,
        image_shape=(batch_size, nkerns[2], 16, 16),
        filter_shape=(nkerns[2], nkerns[2], 3, 3),
        poolsize=(0.5, 0.5)
    )
    
    add1 = T.concatenate([layer3.output,layer6.output],axis=1)
    
    
    #there may be type error
    upsampling2 = T.extra_ops.repeat(add1, 2, axis=2)
    upsampling2 = T.extra_ops.repeat(upsampling2, 2, axis=3)
    
    layer7 = ConvLayer(
        rng,
        input = upsampling2,
        image_shape=(batch_size, nkerns[3], 32, 32),
        filter_shape=(nkerns[1], nkerns[3], 3, 3),
        poolsize=(1, 1)
    )
    layer8 = ConvLayer(
        rng,
        input = layer7.output,
        image_shape=(batch_size, nkerns[1], 32, 32),
        filter_shape=(nkerns[1], nkerns[1], 3, 3),
        poolsize=(1, 1)
    )
    add2 = T.concatenate([layer1.output,layer8.output],axis=1)
    
    layer9 = ConvLayer(
        rng,
        input = add2,
        image_shape=(batch_size, nkerns[2], 32, 32),
        filter_shape=(3, nkerns[2], 3, 3),
        poolsize=(1, 1)
    )
    
    

    outputlayer = outputlayerclass(input=layer9.output)
    cost = outputlayer.mse(y.reshape((batch_size, 3, 32, 32)))

    #downsampling and upsampling params has already been included with the pseudo-poolsize
    test_model = theano.function(
        [index],
        outputlayer.errors(y.reshape((batch_size, 3, 32, 32))),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    validate_model = theano.function(
        [index],
        outputlayer.errors(y.reshape((batch_size, 3, 32, 32))),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    params = layer0.params + layer1.params + layer2.params + layer3.params + layer4.params + layer5.params + layer6.params + layer7.params + layer8.params + layer9.params
    

    updates = adam(cost,params)
    
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    final_test_model = theano.function(
        [index],
        layer9.output,
        givens={
            x: test_set_x[:index],
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.85  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        
        for minibatch_index in range(n_train_batches):
            


            iter = (epoch - 1) * n_train_batches + minibatch_index
            
            if iter % 100 == 0:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f ' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f ') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
            
            if patience <= iter:
                done_looping = True
                break
    
    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f  obtained at iteration %i, '
          'with test performance %f ' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))
    
    

    
    restored_test_set = final_test_model(batch_size)
    output = [not_shared_test_set_y,not_shared_test_set_x,restored_test_set]
    
    return output
    
    

