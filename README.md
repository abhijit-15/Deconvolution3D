# Deconvolution3D
Codes for 3D Deconvolution of Microscopy Images

EE367 Final Project
3D Deconvolution with Deep Learning
Abhijeet Phatak (aphatak@stanford.edu)


## Getting Started

1.36797.tif is the dataset as obtained from www.cellimagelibrary.org/images/36797

2.The dataset is 1904x1900x88 images. 44 widefield [WF] (observed/blurred), 44 structured illumination [SIM] (ground truth)
PSF is estimated with deconvblind function with 20 iterations. More iterations might give better results but please use edgetaper if so otherwise there might be ringing effects. 

3.Read TIPS section https://www.mathworks.com/help/images/ref/deconvblind.html

4.First run getPSF.m which will use MATLAB’s blind deconvolution algorithm to generate the PSF. It also scales the stack laterally to make computation faster.

5.It also stores different stacks in .mat format so that they can be used directly by the different methods.

### RL
1. Ensure that you have run getPSF.m
2. Then run runRL.m

## ADMM
1. Ensure that you have run getPSF.m
2. Then run runADMM.m

##

6. For detailed understanding of the above two methods you may want to refer this paper.
(https://www.osapublishing.org/abstract.cfm?uri=AIO-2016-JT3A.44)

7.The matrices are stored in MATLAB v6 binary format for ease of import into python without loss of performance.

## SRCNN
1. Please ensure that the tensorflow-gpu is installed. Otherwise training will take lot more time.
2. Go to srcnn folder. Run main.py.
3. helper.py contains some auxiliary functions. model.py is the main code for the SRCNN model.
4. For measuring the effectiveness, we train on 22 slices with WF as input and SIM as output.
5. Testing is done on the remaining 22 slices.
6. Please refer to the paper for further details.(http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)
7. Padding is done so as to maintain only one image size.

## Isonet
1. Please ensure that the tensorflow-gpu is installed. Otherwise training will take lot more time.
2. Go to isonet folder. Run main.py.
3. Fiji/ImageJ is also needed to perform some basic image corrections and thresholding operations.  
4. Please refer to the paper for further details. (https://arxiv.org/pdf/1704.01510.pdf)
5. Isonet-1 and 2 are under improvement and testing
