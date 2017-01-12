import os
from os import walk
import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy.linalg as linalg
from natsort import natsorted

from PIL import Image

import theano
import theano.tensor as T
from theano import shared
from theano import function
from theano import shared
from theano.tensor.shared_randomstreams import RandomStreams

'''
Implement the functions that were not implemented and complete the
parts of main according to the instructions in comments.
'''


def reconstructed_image(D, c, num_coeffs, X_mean, im_num):
    '''
    This function reconstructs an image given the number of
    coefficients for each image specified by num_coeffs
    '''

    '''
        Parameters
    ---------------
    c: np.ndarray
        a n x m matrix  representing the coefficients of all the image blocks.
        n represents the maximum dimension of the PCA space.
        m is (number of images x n_blocks**2)

    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)

    im_num: Integer
        index of the image to visualize

    X_mean: np.ndarray
        a matrix representing the mean block.

    num_coeffs: Integer
        an integer that specifies the number of top components to be
        considered while reconstructing
    '''

    c_im = c[:num_coeffs, im_num]
    D_im = D[:, :num_coeffs]
    pic = c_im.T.dot(D_im.T)
    pic.resize(256,256)
    X_recon_img = pic
    


    return X_recon_img


def plot_reconstructions(D, c, num_coeff_array, X_mean, im_num):
    '''
    Plots 9 reconstructions of a particular image using D as the basis matrix and coeffiecient
    vectors from c

    Parameters
    ------------------------
        num_coeff_array: Iterable
            an iterable with 9 elements representing the number_of coefficients
            to use for reconstruction for each of the 9 plots

        c: np.ndarray
            a l x m matrix  representing the coefficients of all blocks in a particular image
            l represents the dimension of the PCA space used for reconstruction
            m represents the number of blocks in an image

        D: np.ndarray
            an N x l matrix representing l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in a block)

        X_mean: basis vectors represent the divergence from the mean so this
            matrix should be added to all reconstructed blocks

        im_num: Integer
            index of the image to visualize
    '''
    f, axarr = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            plt.axes(axarr[i, j])
            plt.imshow(reconstructed_image(D, c, num_coeff_array[i * 3 + j], X_mean, im_num), cmap=cm.Greys_r)

    f.savefig('output/hw1b_{0}.png'.format(im_num))
    plt.close(f)


def plot_top_16(D, sz, imname):
    '''
    Plots the top 16 components from the basis matrix D.
    Each basis vector represents an image block of shape (sz, sz)

    Parameters
    -------------
    D: np.ndarray
        N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)
        n represents the maximum dimension of the PCA space (assumed to be atleast 16)

    sz: Integer
        The height and width of each block

    imname: string
        name of file where image will be saved.
    '''
    # TODO: Obtain top 16 components of D and plot them

    f, axarr = plt.subplots(4,4)
    for i in range(4):
        for j in range(4):
            plt.axes(axarr[i,j])
            plt.imshow(D[:,i+j].reshape(sz,sz), cmap=cm.Greys_r)
    f.savefig(imname)
    plt.close(f)
    print('finish')


def main():
    '''
    Read here all images(grayscale) from Fei_256 folder and collapse
    each image to get an numpy array Ims with size (no_images, height*width).
    Make sure the images are read after sorting the filenames
    '''
    #TODO: Write a code snippet that performs as indicated in the above comment
    '''
    '''
    for root, dirs, files in os.walk("Fei_256"):
        pass
    
    A = [f for f in files if f.endswith(".jpg")]
    A = natsorted(A, key=lambda y: y.lower())
    no_images = len(A)


    im = Image.open("Fei_256/image0.jpg")
    (height, width)= im.size
    size = height*width
    

    Ims = np.zeros((no_images, height*width))
    imPath = ["" for i in range(no_images)]
    for i in range(no_images):
        imPath[i] = "Fei_256/" + A[i]
        img = Image.open(imPath[i])
        arr = np.array(img)
        for j in range(height*width):
            Ims[i,j]=arr.reshape(height*width)[j]

    print(Ims.shape)

    Ims = Ims.astype(np.float32)
    X_mn = np.mean(Ims, 0)
    X = Ims - np.repeat(X_mn.reshape(1, -1), Ims.shape[0], 0)
    print(X.shape)
    

    '''
    Use theano to perform gradient descent to get top 16 PCA components of X
    Put them into a matrix D with decreasing order of eigenvalues

    If you are not using the provided AMI and get an error "Cannot construct a ufunc with more than 32 operands" :
    You need to perform a patch to theano from this pull(https://github.com/Theano/Theano/pull/3532)
    Alternatively you can downgrade numpy to 1.9.3, scipy to 0.15.1, matplotlib to 1.4.2
    '''

     #TODO: Write a code snippet that performs as indicated in the above comment

    #ls = [0.]
    #ds = [np.zeros((1, 256 ** 2))]
    #x = T.dmatrix('x')
    #d = T.dvector('d')
    #y = T.dvector('y')

    #le = d.T.dot(x.T).dot(x).dot(d)
    #ye0 = d + rate * T.grad(le, d)


    #ye1 = d + rate * T.grad(le - (ls[-1] * d.T.dot(ds[-1]).dot(ds[-1].T).dot(d)), d)
    #de = y / y.norm(1)


    #df = function([y], de)
    #lf = function([d, x], le)
    #yf0 = function([d, x], ye0)
    #yf1 = function([d, x], ye1)
   
    Xenvals = np.ones(16)
    Xenvecs = np.ones((256**2,16))

    for i in range(16):

        sumdd = 0.0
        rate = 0.0001
        XX = T.matrix('XX')
        dinit = np.random.rand(256**2)
        d1 = dinit / linalg.norm(dinit)
        d = shared(d1)
        Xenvals[i] = d1.T.dot(X.T).dot(X).dot(d1)
        Xenvecs[:,i] = d1
        
        #manipulating the cost function to take out the already determined components
        j = 0
        for j in range(i):
            sumdd += Xenvals[i]
        sumdd = sumdd*(d.get_value().T.dot((Xenvecs[:,i]).dot(Xenvecs[:,i].T)).dot(d.get_value()))
        
        if i == 0:
            cost = d.T.dot(XX.T).dot(XX).dot(d)
            print(cost.shape)
        if i != 0:
            cost = d.T.dot(XX.T).dot(XX).dot(d) - sumdd
            print(sumdd.shape)
        print(sumdd)
        
        y = d + rate * T.grad(cost, d)
        update = y/y.norm(1)
        t = 1
        updatedy = function([XX], [], updates = ((d, update),))

        
        while t < 50:
            updatedy(X)
            t += 1
        Xenvals[i] = d.get_value().T.dot(X.T).dot(X).dot(d.get_value())
        Xenvecs[:,i] = d.get_value()
    D = Xenvecs
    c = np.dot(D.T, X.T)


    for i in range(0, 200, 10):
        plot_reconstructions(D=D, c=c, num_coeff_array=[1, 2, 4, 6, 8, 10, 12, 14, 16],
                             X_mean=X_mn.reshape((256, 256)), im_num=i)

    plot_top_16(D, 256, 'output/hw1b_top16_256.png')


if __name__ == '__main__':
    main()

