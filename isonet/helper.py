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
