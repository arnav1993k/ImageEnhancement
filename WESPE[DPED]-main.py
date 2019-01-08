#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import os
import scipy.misc
from easydict import EasyDict as edict
from WESPE import *
from utils import *
from dataloader.dataloader import *
from ops import *

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

config = edict()
# training parameters
config.batch_size = 32
config.patch_size = 100
config.mode = "RGB"
config.channels = 3
config.content_layer = 'relu2_2' # originally relu5_4 in DPED
config.learning_rate = 1e-4
config.augmentation = True #data augmentation (flip, rotation)
config.test_every = 500
config.train_iter = 20000

# weights for loss
config.w_content = 0.1 # reconstruction (originally 1)
config.w_color = 20 # gan color (originally 5e-3)
config.w_texture = 3 # gan texture (originally 5e-3)
config.w_tv = 1/400 # total variation (originally 400)

config.model_name = "WESPE_DPED"
pre_path = "D:/Arnub/WESPE/dped/dped/dped"
# directories
config.dataset_name = "iphone"
config.train_path_phone = os.path.join(pre_path,str(config.dataset_name),"training_data",str(config.dataset_name),"*.jpg")
config.train_path_dslr = os.path.join(pre_path,str(config.dataset_name),"training_data/canon/*.jpg")
config.test_path_phone_patch = os.path.join(pre_path,str(config.dataset_name),"test_data/patches",str(config.dataset_name),"*.jpg")
config.test_path_dslr_patch = os.path.join(pre_path,str(config.dataset_name),"test_data/patches/canon/*.jpg")
# config.test_path_phone_image = os.path.join("/Users/arnavkundu/Analytics/DPED/dped_old/test_photos", str(config.dataset_name),"*.jpg")
# config.test_path_dslr_image = os.path.join("/Users/arnavkundu/Studies/Interview/dped\ 2/iphone/test_data/full_size_test_images/canon/*.jpg")

config.vgg_dir = "../vgg_pretrained/imagenet-vgg-verydeep-19.mat"

config.result_dir = os.path.join("./result", config.model_name)
config.result_img_dir = os.path.join(config.result_dir, "samples")
config.checkpoint_dir = os.path.join(config.result_dir, "model")

if not os.path.exists(config.checkpoint_dir):
    print("creating dir...", config.checkpoint_dir)
    os.makedirs(config.checkpoint_dir)
if not os.path.exists(config.result_dir):
    print("creating dir...", config.result_dir)
    os.makedirs(config.result_dir)
if not os.path.exists(config.result_img_dir):
    print("creating dir...", config.result_img_dir)
    os.makedirs(config.result_img_dir)
    
config.sample_dir = "samples"
if not os.path.exists(config.sample_dir):
    print("creating dir...", config.sample_dir)
    os.makedirs(config.sample_dir)


# In[ ]:


# load dataset
dataset_phone, dataset_dslr = load_dataset(config)


# In[ ]:


# build WESPE model
tf.reset_default_graph()
# uncomment this when only trying to test the model
#dataset_phone = []
#dataset_dslr = []
sess = tf.Session()
model = WESPE(sess, config, dataset_phone, dataset_dslr)


# In[ ]:


# train generator & discriminator together
model.train(load = False)


# In[ ]:


# test trained model
model.test_generator(200, 14, load = True)


# In[ ]:


# save trained model
model.save()


# In[ ]:


import numpy as np
a = np.random.randn(3,4,1)
b= np.random.randn(3,1,2)
c = np.random.randn(3,1,1)
d = c+a*b
print(d.shape)

