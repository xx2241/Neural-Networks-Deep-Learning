import numpy
import sys
import os
import timeit
import scipy.ndimage
import theano
import theano.tensor as T
import numpy as np
import random
from utils import floatX
from theano.tensor.signal import downsample
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
from collections import OrderedDict

class Convlayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, border_mode='half',poolsize=(1, 1),values= numpy.asarray(
               np.zeros((1,2,3,4)),
                dtype=theano.config.floatX
            )):
        """
        Allocate a LeNetConvPoolLayer withared variable internal parameters.

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
               values,
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
            border_mode=border_mode,
        )
        self.output = T.nnet.relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input




    
def build_model(x,rng,Weight):
    #x=T.tensor4('x') 

    conv1_input = x.reshape((1,3,600,600))

    net={}
    


    net['conv1_1']=Convlayer(rng,input = conv1_input,image_shape=(1,3,600,600),filter_shape=(64,3,3,3),poolsize=(1,1),values=Weight[0])

    net['conv1_2']=Convlayer(rng,input=net['conv1_1'].output,image_shape=
(1,64,600,600),filter_shape=(64,64,3,3),poolsize=(1,1),values=Weight[2])

    net['pool1']=pool.pool_2d(input=net['conv1_2'].output,ds=(2,2),ignore_border=True , mode ='average_exc_pad')

    net['conv2_1']=Convlayer(rng,input = net['pool1'],image_shape=
(1,64,300,300),filter_shape=(128,64,3,3),poolsize=(1,1),values=Weight[4])
    net['conv2_2']=Convlayer(rng,input=net['conv2_1'].output,image_shape=
(1,128,300,300),filter_shape=(128,128,3,3),poolsize=(1,1),values=Weight[6])

    net['pool2']=pool.pool_2d(input=net['conv2_2'].output,ds=(2,2),ignore_border=True , mode ='average_exc_pad')

    net['conv3_1']=Convlayer(rng,input=net['pool2'],image_shape=(1,128,150,150),filter_shape=(256,128,3,3),poolsize=(1,1),values=Weight[8])
    net['conv3_2']=Convlayer(rng,input=net['conv3_1'].output,image_shape=
(1,256,150,150),filter_shape=(256,256,3,3),poolsize=(1,1),values=Weight[10])
    net['conv3_3']=Convlayer(rng,input=net['conv3_2'].output,image_shape=
(1,256,150,150),filter_shape=(256,256,3,3),poolsize=(1,1),values=Weight[12])
    net['conv3_4']=Convlayer(rng,input=net['conv3_3'].output,image_shape=
(1,256,150,150),filter_shape=(256,256,3,3),poolsize=(1,1),values=Weight[14])

    net['pool3']=pool.pool_2d(input=net['conv3_4'].output,ds=(2,2),ignore_border=True, mode ='average_exc_pad')
 
    net['conv4_1']=Convlayer(rng,input=net['pool3'],image_shape=
(1,256,75,75),filter_shape=(512,256,3,3),poolsize=(1,1),values=Weight[16])
    net['conv4_2']=Convlayer(rng,input=net['conv4_1'].output,image_shape=
(1,512,75,75),filter_shape=(512,512,3,3),poolsize=(1,1),values=Weight[18])
    net['conv4_3']=Convlayer(rng,input=net['conv4_2'].output,image_shape=
(1,512,75,75),filter_shape=(512,512,3,3),poolsize=(1,1),values=Weight[20])
    net['conv4_4']=Convlayer(rng,input=net['conv4_3'].output,image_shape=
(1,512,75,75),filter_shape=(512,512,3,3),poolsize=(1,1),values=Weight[22])

    net['pool4']=pool.pool_2d(input=net['conv4_4'].output,ds=(2,2),ignore_border=True, mode ='average_exc_pad')

    net['conv5_1']=Convlayer(rng,input=net['pool4'],image_shape=(1,512,37,37),filter_shape=(512,512,3,3),poolsize=(1,1),values=Weight[24])
    net['conv5_2']=Convlayer(rng,input=net['conv5_1'].output,image_shape=
(1,512,37,37),filter_shape=(512,512,3,3),poolsize=(1,1),values=Weight[26])
    net['conv5_3']=Convlayer(rng,input=net['conv5_2'].output,image_shape=
(1,512,37,37),filter_shape=(512,512,3,3),poolsize=(1,1),values=Weight[28])
    net['conv5_4']=Convlayer(rng,input=net['conv5_3'].output,image_shape=(
1,512,37,37),filter_shape=(512,512,3,3),poolsize=(1,1),values=Weight[30])

    net['pool5']=pool.pool_2d(input=net['conv5_4'].output,ds=(2,2),ignore_border=True, mode ='average_exc_pad')
    
    
    return net

