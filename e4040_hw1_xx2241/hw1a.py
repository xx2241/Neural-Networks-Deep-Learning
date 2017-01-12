import os
from os import walk
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy.linalg as linalg
from natsort import natsorted
from theano.tensor.nnet.neighbours import images2neibs
from theano.tensor.nnet.neighbours import neibs2images

from PIL import Image

import theano
import theano.tensor as T


'''
Implement the functions that were not implemented and complete the
parts of main according to the instructions in comments.
'''

def reconstructed_image(D,c,num_coeffs,X_mean,n_blocks,im_num):
    '''
    This function reconstructs an image X_recon_img given the number of
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
        

    n_blocks: Integer
        number of blocks comprising the image in each direction.
        For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4
    '''

    
    c_im = c[:num_coeffs,n_blocks*n_blocks*im_num:n_blocks*n_blocks*(im_num+1)]
    D_im = D[:,:num_coeffs]
    print(c_im.shape)
    print(D_im.shape)
    p = np.dot(c_im.T, D_im.T)
    print(p.shape)
    #print(im_val.shape)
    #print(X.shape)
    #TODO: Enter code below for reconstructing the image X_recon_img
    
    sz = int(256/n_blocks)
    Q = np.repeat(X_mean, n_blocks, 0)
    print(sz)
    print(Q.shape)
    P = np.repeat(Q, n_blocks, 1)
    print(P.shape)
    neibs = T.matrix('neibs')
    im_new = neibs2images(neibs, (sz,sz), (256, 256))
    inv_window = theano.function([neibs], im_new)
    X_recon_img = inv_window(p) + P
    return X_recon_img

def plot_reconstructions(D,c,num_coeff_array,X_mean,n_blocks,im_num):
    '''
    Plots 9 reconstructions of a particular image using D as the basis matrix and coeffiecient
    vectors from c

    Parameters
    ------------------------
        num_coeff_array: Iterable
            an iterable with 9 elements representing the number of coefficients
            to use for reconstruction for each of the 9 plots
        
        c: np.ndarray
            a l x m matrix  representing the coefficients of all blocks in a particular image
            l represents the dimension of the PCA space used for reconstruction
            m represents the number of blocks in an image

        D: np.ndarray
            an N x l matrix representing l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in a block)

        n_blocks: Integer
            number of blocks comprising the image in each direction.
            For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4

        X_mean: basis vectors represent the divergence from the mean so this
            matrix should be added to all reconstructed blocks

        im_num: Integer
            index of the image to visualize
    '''
    f, axarr = plt.subplots(3,3)
    for i in range(3):
        for j in range(3):
            plt.axes(axarr[i,j])
            plt.imshow(reconstructed_image(D,c,num_coeff_array[i*3+j],X_mean,n_blocks,im_num))
            
    f.savefig('output/hw1a_{0}_im{1}.png'.format(n_blocks, im_num))
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
    #N = D
    #N.reshape((8,8))
    f, axarr = plt.subplots(4,4)
    for i in range(4):
        for j in range(4):
            plt.axes(axarr[i,j])
            plt.imshow(D[:,i+j].reshape(sz,sz))
    f.savefig(imname)
    plt.close(f)

    #TODO: Obtain top 16 components of D anod plot them
    
    #raise NotImplementedError

    
def main():
    '''
    Read here all images(grayscale) from Fei_256 folder
    into an numpy array Ims with size (no_images, height, width).
    Make sure the images are read after sorting the filenames
    '''
    for root, dirs, files in os.walk("Fei_256"):
        pass
        #no_images = len(files)

    A = [f for f in files if f.endswith(".jpg")]
    A = natsorted(A, key=lambda y: y.lower())
    no_images = len(A)
    
    im = Image.open("Fei_256/image0.jpg")
    (height, width)= im.size
    imArray = np.zeros((no_images, 1, height, width))
    imPath = ["" for i in range(no_images)]

    for i in range(no_images):
        imPath[i] = "Fei_256/" + A[i]
        imArray[i, 0,:,:] = Image.open(imPath[i])
    #TODO: Read all images into a numpy array of size (no_images, height, width)
    
    szs = [8, 32, 64]
    num_coeffs = [range(1, 10, 1), range(3, 30, 3), range(5, 50, 5)]

    for sz, nc in zip(szs, num_coeffs):
        '''
        Divide here each image into non-overlapping blocks of shape (sz, sz).
        Flatten each block and arrange all the blocks in a
        (no_images*n_blocks_in_image) x (sz*sz) matrix called X
        ''' 
        images = T.tensor4('images')
        neibs = images2neibs(images, neib_shape=(sz, sz))
        window_function = theano.function([images], neibs)
        im_val = imArray
        X = window_function(im_val)
        #TODO: Write a code snippet that performs as indicated in the above comment


        X_mean = np.mean(X, 0)
        X = X - np.repeat(X_mean.reshape(1, -1), X.shape[0], 0)
        print(X_mean.shape)
        print(X.shape)
        print(np.repeat(X_mean.reshape(1, -1), X.shape[0], 0).shape)
        print(X_mean.reshape(1,-1).shape)
        '''
        Perform eigendecomposition on X^T X and arrange the eigenvectors
        in decreasing order of eigenvalues into a matrix D
        '''
        covMat = np.cov(X, rowvar=0)
        eigVals, eigVects = np.linalg.eigh(X.T.dot(X))
        D = eigVects[:, ::-1]
        #eigVals, eigVects = np.linalg.eigh(np.mat(covMat))
        #eigValindice = np.argsort(eigVals)
        #n_eigValindice = eigValindice[-1:-(n+1):-1]
        #n_eigVect = eigVects[:,eigValindice]
        #D = n_eigVect
        #print(X_mean)
        print(X.shape[0])
        #print(Y)
        print(D.shape)
        #print(eigVals)
        #print(eigVects)



        #TODO: Write a code snippet that performs as indicated in the above comment
        






        c = np.dot(D.T, X.T)
        for i in range(0, 200, 10):
           plot_reconstructions(D=D, c=c, num_coeff_array=nc, X_mean=X_mean.reshape((sz, sz)), n_blocks=int(256/sz), im_num=i)

        plot_top_16(D, sz, imname='output/hw1a_top16_{0}.png'.format(sz))


if __name__ == '__main__':
    main()

