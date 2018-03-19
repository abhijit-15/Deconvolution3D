import os
import numpy as np
import scipy.io as spio
from tifffile import imsave as tiffsave
from tifffile import imread as tiffread
from skimage.measure import compare_psnr

def read_from_mat(file):
    '''Used to read 3D stacks containing PSF, OTF or data from MATLAB
    which was stored in v6 binary format. Output is a numpy ndarray.'''
    mat = spio.loadmat(file , squeeze_me=True)
    return mat

def save_stack_as_images(arr):
    sx , sy , sz = arr.shape
    for i in range(sz):
        tiffsave(str(i)+'.tif' , data=arr[:,:,i] , dtype=np.float32)
    print('Completed saving stack'),

path = os.getcwd()

observed = read_from_mat('stack.mat')['stack']
ground  = read_from_mat('ground.mat')['ground']

train_data_in = observed[:,:,0:22]
train_data_out = ground[:,:,0:22]

test_data_in = observed[:,:,23:]
test_data_out = ground[:,:,23:]
