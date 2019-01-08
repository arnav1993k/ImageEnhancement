from __future__ import division
import time
import tensorflow as tf
#import tensorlayer as tl
import scipy.misc
import scipy.io
import numpy as np
from utils import *
from ops import *

def resblock(feature_in, num):
    # subblock (conv. + BN + relu)
    temp =  tf.layers.conv2d(feature_in, 32, 3, strides = 1, padding = 'SAME', name = ('resblock_%d_CONV_1' %num), kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
    # temp = tf.layers.batch_normalization(temp, name = ('resblock_%d_BN_1' %num))
    temp = tf.nn.relu(temp)
        
    # subblock (conv. + BN + relu)
    temp =  tf.layers.conv2d(temp, 32, 3, strides = 1, padding = 'SAME', name = ('resblock_%d_CONV_2' %num), kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
    # temp = tf.layers.batch_normalization(temp, name = ('resblock_%d_BN_2' %num))
    temp = tf.nn.relu(temp)
    return temp + feature_in
'''
b1 = conv_block(input, 384, 1, 1)

    b2 = conv_block(input, 192, 1, 1)
    b2 = conv_block(b2, 224, 1, 7)
    b2 = conv_block(b2, 256, 7, 1)

    b3 = conv_block(input, 192, 1, 1)
    b3 = conv_block(b3, 192, 7, 1)
    b3 = conv_block(b3, 224, 1, 7)
    b3 = conv_block(b3, 224, 7, 1)
    b3 = conv_block(b3, 256, 1, 7)

    b4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
    b4 = conv_block(b4, 128, 1, 1)

    m = merge([b1, b2, b3, b4], mode='concat', concat_axis=channel_axis)
    return m
# '''
# def inception_blockA(fan_in,num):

def inception_block(feature_in,num):
    simple1 =  tf.layers.conv2d(feature_in, 64, 1, strides = 1, padding = 'SAME', name = ('inceptionblock_%d_CONV_1x1_1' %num), kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
    simple2 =  tf.layers.conv2d(feature_in, 64, 1, strides = 1, padding = 'SAME', name = ('inceptionblock_%d_CONV_1x1_2' %num), kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
    simple3 =  tf.layers.conv2d(feature_in, 64, 1, strides = 1, padding = 'SAME', name = ('inceptionblock_%d_CONV_1x1_3' %num), kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE, activation='relu')
    filter1 = tf.layers.conv2d(simple1, 32, 3, strides = 1, padding = 'SAME', name = ('inceptionblock_%d_CONV_3x3' %num), kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE, activation='relu')
    filter2 = tf.layers.conv2d(simple2, 32, 5, strides = 1, padding = 'SAME', name = ('inceptionblock_%d_CONV_5x5' %num), kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE, activation='relu')
    stack = tf.concat(axis=3, values=[simple3, filter1, filter2,feature_in])
    return stack

def generator_network(image, var_scope = 'generator'):
    with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
        # conv. layer before residual blocks 
        # b1_in = tf.layers.conv2d(image, 100, 15, strides = 1, padding = 'SAME', name = 'CONV_1', kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        # b1_in = tf.nn.relu(b1_in)
            
        # # residual blocks
        # b1_out = resblock(b1_in, 1)
        # b2_out = resblock(b1_out, 2)
        # b3_out = resblock(b2_out, 3)
        # b4_out = resblock(b3_out, 4)
        temp = inception_block(image,1)
        # # conv. layers after residual blocks
        temp = tf.layers.conv2d(temp, 64, 3, strides = 1, padding = 'SAME', name = 'CONV_2', kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        temp = tf.nn.relu(temp)
        # temp = tf.layers.conv2d(temp, 64, 3, strides = 1, padding = 'SAME', name = 'CONV_3', kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        # temp = tf.nn.relu(temp)
        # temp = tf.layers.conv2d(temp, 64, 3, strides = 1, padding = 'SAME', name = 'CONV_4', kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        # temp = tf.nn.relu(temp)
        temp = tf.layers.conv2d(temp, 3, 1, strides = 1, padding = 'SAME', name = 'CONV_5', kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        return temp

def discriminator_network(image, var_scope = 'discriminator', preprocess = 'gray'):
    with tf.variable_scope(var_scope, reuse = tf.AUTO_REUSE):
        if preprocess == 'gray':
            #convert to grayscale image
            print("Discriminator-texture")
            image_processed = tf.image.rgb_to_grayscale(image)
        elif preprocess == 'blur':
            print("Discriminator-color (blur)")
            image_processed = gaussian_blur(image)
        else:
            print("Discriminator-color (none)")
            image_processed = image
            
        # conv layer 1 
        temp = tf.layers.conv2d(image_processed, 48, 11, strides = 4, padding = 'SAME', name = 'CONV_1', kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        temp = lrelu(temp)
            
        # conv layer 2
        temp = tf.layers.conv2d(temp, 128, 5, strides = 2, padding = 'SAME', name = 'CONV_2', kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        temp = tf.layers.batch_normalization(temp, name = 'BN_2')
        temp = lrelu(temp)
            
        # conv layer 3
        temp = tf.layers.conv2d(temp, 192, 3, strides = 1, padding = 'SAME', name = 'CONV_3', kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        temp = tf.layers.batch_normalization(temp, name = 'BN_3')
        temp = lrelu(temp)
            
        # conv layer 4
        temp = tf.layers.conv2d(temp, 192, 3, strides = 1, padding = 'SAME', name = 'CONV_4', kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        temp = tf.layers.batch_normalization(temp, name = 'BN_4')
        temp = lrelu(temp)
            
        # conv layer 5
        temp = tf.layers.conv2d(temp, 128, 3, strides = 2, padding = 'SAME', name = 'CONV_5', kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        temp = tf.layers.batch_normalization(temp, name = 'BN_5')
        temp = lrelu(temp)
            
        # FC layer 1
        fc_in = tf.contrib.layers.flatten(temp)
        fc_out = tf.layers.dense(fc_in, units = 1024, activation = None)
        fc_out = lrelu(fc_out)
            
        # FC layer 2
        logits = tf.layers.dense(fc_out, units = 1, activation = None)
        probability = tf.nn.sigmoid(logits)
    return logits, probability