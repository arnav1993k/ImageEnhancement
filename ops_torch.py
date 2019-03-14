from __future__ import division
import torch
import numpy as np
import scipy.stats as st
import cv2
import torch.nn.functional as F

mean_RGB = np.array([123.68, 116.779, 103.939])

def preprocess(img):
    return (img - mean_RGB)/255 

def postprocess(img):
    return np.round(np.clip(img*255 + mean_RGB, 0, 255)).astype(np.uint8)

    
def gauss_kernel(kernlen=21, nsig=3, channels=1):
    
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis = 2)
    
    return out_filter

def gaussian_blur(x):
    
    kernel_var = gauss_kernel(21, 3, 3)
    kernel_var = np.transpose(kernel_var, (2,3,0,1))
    kernel_var = torch.from_numpy(kernel_var)
    kernel_var = kernel_var.float()
    
    if torch.cuda.is_available():
        kernel_var = kernel_var.cuda()

    return F.conv2d(x, kernel_var, padding=10, groups=3)

def resize(x,n):
    return cv2.resize(x,None,fx=n, fy=n, interpolation = cv2.INTER_CUBIC)