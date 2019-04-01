#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:26:44 2019

@author: vineet
"""

from __future__ import division
import numpy as np
import scipy.io
import torch.nn.functional as F
import torch

def net(path_to_vgg_net, input_image):

    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    data = scipy.io.loadmat(path_to_vgg_net)
    weights = data['layers'][0]

    net = {}
    current = input_image
    for i, name in enumerate(layers):
        layer_type = name[:4]
        if layer_type == 'conv':
            kernels, bias = weights[i][0][0][2][0]
            
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            kernels = np.transpose(kernels, (3, 2, 0, 1))
            
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias)
        elif layer_type == 'relu':
            current = F.relu(current)
        elif layer_type == 'pool':
            current = _pool_layer(current)
        net[name] = current

    return net


def _conv_layer(input, weights, bias):
    
    weights = torch.from_numpy(weights)
    weights = weights.float()
    if torch.cuda.is_available():
        weights = weights.cuda()

    kernel_shape = weights.shape[2]
    conv = F.conv2d(input, weights, stride=1, padding=(kernel_shape-1)//2)
    
    
    bias = torch.from_numpy(bias)
    bias = bias.float()
    bias = bias.cuda()
    bias = bias.unsqueeze(1)
    bias = bias.unsqueeze(2)
    
    return conv + bias    
    
def _pool_layer(input):
    return F.max_pool2d(input, 2, stride=1)