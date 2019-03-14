from __future__ import division
import numpy as np
import time
from glob import glob
import scipy.misc
import scipy.io
from ops_torch import preprocess
import torchvision.transforms as ttransforms


class Dataset(object):
    """docstring for Dataset"""
    def __init__(self, config):
        super(Dataset, self).__init__()

        self.config = config
        self.phone_list = np.array(sorted(glob(config.train_path_phone)))
        self.canon_list = np.array(sorted(glob(config.train_path_canon)))
        self.DIV2K_list = np.array(sorted(glob(config.train_path_DIV2K)))

        print("Dataset: %s, %d images" %(self.config.dataset_name, len(self.phone_list)))
        print("DIV2K: %d images" %(len(self.DIV2K_list)))
        
        start_time = time.time()
        # self.dataset_phone = [scipy.misc.imread(filename, mode = "RGB") for filename in self.phone_list]
        # self.dataset_canon = [scipy.misc.imread(filename, mode = "RGB") for filename in self.canon_list]
        # self.dataset_DIV2K = [scipy.misc.imread(filename, mode = "RGB") for filename in self.DIV2K_list]

        self.dataset_phone = [filename for filename in self.phone_list]
        self.dataset_canon = [filename for filename in self.canon_list]
        self.dataset_DIV2K = [filename for filename in self.DIV2K_list]

        print("%d images loaded! setting took: %4.4fs" % (len(self.dataset_phone), time.time() - start_time))

        self.len_dataset_phone = len(self.dataset_phone)
        self.len_dataset_canon = len(self.dataset_canon)
        self.len_dataset_DIV2K = len(self.dataset_DIV2K)


    def __getitem__(self, idx):

        # phone_img = self.dataset_phone[idx]
        # canon_img = self.dataset_canon[idx]
        # DIV2K_img = self.dataset_DIV2K[idx]

        phone_img = scipy.misc.imread(self.dataset_phone[idx], mode = "RGB")
        canon_img = scipy.misc.imread(self.dataset_canon[idx], mode = "RGB")

        index = np.random.randint(self.len_dataset_DIV2K)
        DIV2K_img = scipy.misc.imread(self.dataset_DIV2K[index], mode = "RGB")

        patch_size = self.config.patch_size
        H = 0
        W = 0
        if phone_img.shape[0]>100:
            H = np.random.randint(phone_img.shape[0]-patch_size)
            W = np.random.randint(phone_img.shape[1]-patch_size)
        
        phone_patch = phone_img[H: H + patch_size, W: W + patch_size, :]
        canon_patch = canon_img[H: H + patch_size, W: W + patch_size, :]

        H = np.random.randint(DIV2K_img.shape[0] - patch_size)
        W = np.random.randint(DIV2K_img.shape[1] - patch_size)
        
        DIV2K_patch = DIV2K_img[H: H + patch_size, W: W + patch_size, :]

        if self.config.augmentation == True:
            
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

        phone_patch = preprocess(phone_patch) # pre/post processing function is defined in ops.py
        canon_patch = preprocess(canon_patch)
        DIV2K_patch = preprocess(DIV2K_patch)
        
        phone_patch = self._img_transform(phone_patch)
        canon_patch = self._img_transform(canon_patch)
        DIV2K_patch = self._img_transform(DIV2K_patch)

        return phone_patch, canon_patch, DIV2K_patch
    

    def __len__ (self):
        return self.len_dataset_phone
    
    
    def _img_transform(self, image):
        
        image_transforms = ttransforms.Compose([
            ttransforms.ToTensor(),
        ])
        image = image_transforms(image)
        return image
    
#    
#class Dataset2(object):
#    """docstring for Dataset"""
#    def __init__(self, config):
#        super(Dataset2, self).__init__()
#
#        self.config = config
#        self.phone_list = np.array(sorted(glob(config.train_path_phone)))
#        self.canon_list = np.array(sorted(glob(config.train_path_canon)))
#        self.DIV2K_list = np.array(sorted(glob(config.train_path_DIV2K)))
#
#        print("Dataset: %s, %d images" %(self.config.dataset_name, len(self.phone_list)))
#        print("DIV2K: %d images" %(len(self.DIV2K_list)))
#        
#        start_time = time.time()
#        self.dataset_phone = [scipy.misc.imread(filename, mode = "RGB") for filename in self.phone_list]
#        self.dataset_canon = [scipy.misc.imread(filename, mode = "RGB") for filename in self.canon_list]
#        self.dataset_DIV2K = [scipy.misc.imread(filename, mode = "RGB") for filename in self.DIV2K_list]
#        print("%d images loaded! setting took: %4.4fs" % (len(self.dataset_phone), time.time() - start_time))
#
#
#    def __getitem__(self, idx):
#
#        phone_img = self.dataset_phone[idx]
#        canon_img = self.dataset_canon[idx]
#        DIV2K_img = self.dataset_DIV2K[idx]
#
#        patch_size = self.config.patch_size
#        H = 0
#        W = 0
#        if phone_img.shape[0]>100:
#            H = np.random.randint(phone_img.shape[0]-patch_size)
#            W = np.random.randint(phone_img.shape[1]-patch_size)
#        
#        phone_patch = phone_img[H: H + patch_size, W: W + patch_size, :]
#        canon_patch = canon_img[H: H + patch_size, W: W + patch_size, :]
#
#        H = np.random.randint(DIV2K_img.shape[0] - patch_size)
#        W = np.random.randint(DIV2K_img.shape[1] - patch_size)
#        
#        DIV2K_patch = DIV2K_img[H: H + patch_size, W: W + patch_size, :]
#
#        if self.config.augmentation == True:
#            
#            prob = np.random.rand()
#            if prob > 0.5:
#                phone_patch = np.flip(phone_patch, axis = 0)
#                canon_patch = np.flip(canon_patch, axis = 0)
#                DIV2K_patch = np.flip(DIV2K_patch, axis = 0)
#                
#            prob = np.random.rand()
#            if prob > 0.5:
#                phone_patch = np.flip(phone_patch, axis = 1)
#                canon_patch = np.flip(canon_patch, axis = 1)
#                DIV2K_patch = np.flip(DIV2K_patch, axis = 1)
#                
#            prob = np.random.rand()
#            if prob > 0.5:
#                phone_patch = np.rot90(phone_patch)
#                canon_patch = np.rot90(canon_patch)
#                DIV2K_patch = np.rot90(DIV2K_patch)
#
#        phone_patch = preprocess(phone_patch) # pre/post processing function is defined in ops.py
#        canon_patch = preprocess(canon_patch)
#        DIV2K_patch = preprocess(DIV2K_patch)
#        
#        phone_patch = self._img_transform(phone_patch)
#        canon_patch = self._img_transform(canon_patch)
#        DIV2K_patch = self._img_transform(DIV2K_patch)
#
#        return phone_patch, canon_patch, DIV2K_patch
#    
#
#    def __len__ (self):
#        return len(self.dataset_phone)
#    
#    
#    def _img_transform(self, image):
#        
#        image_transforms = ttransforms.Compose([
#            ttransforms.ToTensor(),
#        ])
#        image = image_transforms(image)
#        return image







def get_patch(image, patch_size):

    H = np.random.randint(image.shape[0] - patch_size)
    W = np.random.randint(image.shape[1] - patch_size)
    patch = image[H: H + patch_size, W: W + patch_size, :]
    
    return patch
