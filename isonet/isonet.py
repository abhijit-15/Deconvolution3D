'''
    Deconvolution using Deep Learning
    Based on : M.Weigert et.al. "Isotropic reconstruction of 3D Fluorescence Microscopy Images using Convolutional Neural Networks"
'''
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras import backend as K
from keras.losses import mean_squared_error
from scipy.ndimage import zoom
from scipy.ndimage.filters import convolve
import gputools as gt
from scipy.signal import fftconvolve as scipycnv
from gputools.convolve import convolve as cnv
import numpy as np
import tensorflow as tf
from helper import *

def isonet1(shape):
    model = Sequential()
    model.add(Conv2D(filters = 64 , kernel_size = (64,64) , padding = 'same', activation='relu',input_shape=shape))
    model.add(Conv2D(filters = 32 , kernel_size = (5,5), padding = 'same' , activation='relu'))
    model.add(Conv2D(filters = 1 , kernel_size = (5,5), padding = 'same' , activation='relu'))
    model.add(Conv2D(filters = 1 , kernel_size = (1,1), padding = 'same' , activation='relu'))
    return model

def slicing(arr):
    scale = (1 , 1 , 0.5)
    inv_scale = (1 , 1 , 2)
    sliced = zoom(zoom(arr,scale,order=0),inv_scale, order=1)
    return sliced

def convolution_op(stack , psf_rot):
    convolved = []
    for i in range(psf_rot.shape[0]):
        convolved.append(np.real(scipycnv(psf_rot[i,:,:],stack[:,:,i],mode='same')))
    return np.stack(convolved,axis=-1)

def generate_training_samples(stack , ground ,  psf):
    psf_rot = psf.transpose(2,1,0)[...,::-1]
    sx , sy , sz = stack.shape
    p_xy = slicing(convolution_op(ground , psf_rot))
    d = {'x' : [] , 'y' : []}
    for i in range(sz):
        d['y'].append(ground[:,:,i])
        d['x'].append(p_xy[:,:,i])

    d['x'] = np.stack(d['x'],axis=-1)
    d['y'] = np.stack(d['y'],axis=-1)
    return d

def custom_loss(g_xy , grot_xy):
    #Multiple patches
    print(g_xy.shape)
    print(grot_xy.shape)
    #return mean_squared_error(g_xy , grot_xy)
    return -K.mean(20 * K.log(K.max(g_xy))) + 10 * K.log(mean_squared_error(g_xy , tf.transpose(grot_xy,[1,0,2])))

def mean_pred(y_pred):
    return K.mean(y_pred)
