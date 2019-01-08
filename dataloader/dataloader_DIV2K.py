from __future__ import division
import numpy as np
import os
import os.path
import time
from glob import glob
import scipy.misc
import scipy.io
from scipy.misc import imresize

from random import shuffle
import imageio
import cv2
import math
from ops import *
    
def load_dataset(config):
    phone_list = np.array(sorted(glob(config.train_path_phone)))
    canon_list = np.array(sorted(glob(config.train_path_canon)))
    DIV2K_list = np.array(sorted(glob(config.train_path_DIV2K)))
#     indices = np.random.choice(len(phone_list),config.sample_size)
#     phone_list = phone_list[indices]
#     canon_list = canon_list[indices]
    print("Dataset: %s, %d images" %(config.dataset_name, len(phone_list)))
    print("DIV2K: %d images" %(len(DIV2K_list)))
    start_time = time.time()
    dataset_phone = [scipy.misc.imread(filename, mode = "RGB") for filename in phone_list]
    dataset_canon = [scipy.misc.imread(filename, mode = "RGB") for filename in canon_list]
#     dataset_canon = [imresize(img, (dataset_phone[0].shape[0], dataset_phone[0].shape[1])) for img in dataset_canon]
    dataset_DIV2K = [scipy.misc.imread(filename, mode = "RGB") for filename in DIV2K_list]
    print("%d images loaded! setting took: %4.4fs" % (len(dataset_phone), time.time() - start_time))
    return dataset_phone, dataset_canon, dataset_DIV2K

def get_batch(dataset_phone, dataset_canon, dataset_DIV2K=None, config=None, start = 0):
    phone_batch = np.zeros([config.batch_size, config.patch_size, config.patch_size, config.channels], dtype = 'float32')
    canon_batch = np.zeros([config.batch_size, config.patch_size, config.patch_size, config.channels], dtype = 'float32')
    DIV2K_batch = np.zeros([config.batch_size, config.patch_size, config.patch_size, config.channels], dtype = 'float32')

    for i in range(config.batch_size):
        index = np.random.randint(len(dataset_phone))
        phone_img = dataset_phone[index]
        canon_img = dataset_canon[index]
        index = np.random.randint(len(dataset_DIV2K))
        DIV2K_img = dataset_DIV2K[index]
#         print("phone img shape:", phone_img.shape)
#         print("img shape:", DIV2K_img.shape)
        
        patch_size = config.patch_size
        H=0
        W=0
        if phone_img.shape[0]>100:
            H = np.random.randint(phone_img.shape[0]-patch_size)
            W = np.random.randint(phone_img.shape[1]-patch_size)
        phone_patch = phone_img[H: H+patch_size, W: W+patch_size, :]
        canon_patch = canon_img[H: H+patch_size, W: W+patch_size, :]
        H = np.random.randint(DIV2K_img.shape[0]-patch_size)
        W = np.random.randint(DIV2K_img.shape[1]-patch_size)
        DIV2K_patch = DIV2K_img[H: H+patch_size, W: W+patch_size, :]
#         print("phone patch shape:", phone_patch.shape)
#         print("div patch shape:", DIV2K_patch.shape)
        
        #imageio.imwrite("./DIV2Kimg.png", DIV2K_img)
        #imageio.imwrite("./DIV2Kpatch.png", DIV2K_patch)

        # randomly flip, rotate patch (assuming that the patch shape is square)
        if config.augmentation == True:
            prob = np.random.rand()
            if prob > 0.5:
                phone_patch = np.flip(phone_patch, axis = 0)
                canon_patch = np.flip(canon_patch, axis = 0)
                DIV2K_patch = np.flip(DIV2K_patch, axis = 0)
            prob = np.random.rand()
            if prob > 0.5:
                phone_patch = np.flip(phone_patch, axis = 1)
                canon_patch = np.flip(canon_patch, axis = 1)
                DIV2K_patch = np.flip(DIV2K_patch, axis = 1)
            prob = np.random.rand()
            if prob > 0.5:
                phone_patch = np.rot90(phone_patch)
                canon_patch = np.rot90(canon_patch)
                DIV2K_patch = np.rot90(DIV2K_patch)
        #print(index)
        phone_batch[i,:,:,:] = preprocess(phone_patch) # pre/post processing function is defined in ops.py
        canon_batch[i,:,:,:] = preprocess(canon_patch)
        DIV2K_batch[i,:,:,:] = preprocess(DIV2K_patch)
    return phone_batch, canon_batch, DIV2K_batch
def get_patch(image,patch_size):
    H = np.random.randint(image.shape[0]-patch_size)
    W = np.random.randint(image.shape[1]-patch_size)
    phone_patch = image[H: H+patch_size, W: W+patch_size, :]
    return phone_patch