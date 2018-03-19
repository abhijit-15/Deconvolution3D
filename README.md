# Deconvolution3D
Codes for 3D Deconvolution of Microscopy Images

EE367 Final Project
3D Deconvolution with Deep Learning
Abhijeet Phatak

**README**

1.36797.tif is the dataset as obtained from www.cellimagelibrary.org/images/36797

2.The dataset is 1904x1900x88 images. 44 widefield [WF] (observed/blurred), 44 structured illumination [SIM] (ground truth)
PSF is estimated with deconvblind function with 20 iterations. More iterations might give better results but please use edgetaper if so otherwise there might be ringing effects. 

3.Read TIPS section https://www.mathworks.com/help/images/ref/deconvblind.html

4.First run getPSF.m which will use MATLAB’s blind deconvolution algorithm to generate the PSF. It also scales the stack laterally to make computation faster.

5.It also stores different stacks in .mat format so that they can be used directly by the codes.

6.*For RL:*
a.Ensure that you have run getPSF.m
b.Then run runRL.m

7.*For ADMM:*
a.Ensure that you have run getPSF.m
b.Then run runADMM.m

8.For detailed understanding of the above two methods you may want to refer this paper.
(https://www.osapublishing.org/abstract.cfm?uri=AIO-2016-JT3A.44)

9.The matrices are stored in MATLAB v6 binary format for ease of import into python without loss of performance.

10.*For SRCNN:*
a.Please ensure that the tensorflow-gpu is installed. Otherwise training will take lot more time.
b.Go to srcnn folder. Run main.py.
c.helper.py contains some auxiliary functions. model.py is the main code for the SRCNN model.
d.For measuring the effectiveness, we train on 22 slices with WF as input and SIM as output.
e.Testing is done on the remaining 22 slices.
f.Please refer to the paper for further details.(http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)
g.Padding is done so as to maintain only one image size.

11.*For Isonet-1:*
a.Please ensure that the tensorflow-gpu is installed. Otherwise training will take lot more time.
b.Go to isonet folder. Run main.py.
c.Fiji/ImageJ is also needed to perform some basic image corrections and thresholding operations.  
d.Please refer to the paper for further details. (https://arxiv.org/pdf/1704.01510.pdf)
e.Isonet-1 and 2 are under improvement and testing
