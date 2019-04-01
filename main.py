
import os
from easydict import EasyDict as edict
from ImageEnhancement.WESPE_torch import WESPE

config = edict()

# training parameters
config.batch_size = 64#32
config.patch_size = 100
config.mode = "RGB"
config.channels = 3
config.content_layer = 'relu2_2' # originally relu5_4 in DPED
config.learning_rate = 1e-4
config.augmentation = True #data augmentation (flip, rotation)
config.test_every = 200
config.train_iter = 50000
config.data_loader_workers = 4
config.pin_memory = 2
# config.sample_size = 100000

# weights for loss
config.w_content = 2 # reconstruction (originally 1)
config.w_profile = 0.2
config.w_color = 5 # gan color (originally 5e-3)
config.w_texture = 2 # gan texture (originally 5e-3)
config.w_tv = 3 # total variation (originally 400)
config.gamma = 0.6
config.model_name = "WESPE_DIV2K_arnav_gpu1"

# directories
config.dataset_name = "iphone"
config.train_path_phone = os.path.join("/home/grads/v/vineet/Downloads/DPED/dped/iphone/training_data/iphone","*.jpg")
config.train_path_canon = os.path.join("/home/grads/v/vineet/Downloads/DPED/dped/iphone/training_data/canon","*.jpg")
config.train_path_DIV2K = os.path.join("/home/grads/v/vineet/Downloads/DPED/DIV2K_train_HR","*.png")

config.test_path_phone_patch = os.path.join("/home/grads/v/vineet/Downloads/DPED/sample_images/original_images/iphone","*.jpg")
config.test_path_phone_image = os.path.join("/home/grads/v/vineet/Downloads/DPED/sample_images/original_images/iphone","*.jpg")

config.vgg_dir = "./vgg_pretrained/imagenet-vgg-verydeep-19.mat"

config.result_dir = os.path.join("./result_1", config.model_name)
config.result_img_dir = os.path.join(config.result_dir, "samples")
config.checkpoint_dir = os.path.join(config.result_dir, "model")

if not os.path.exists(config.result_dir):
    print("creating dir...", config.result_dir)
    os.makedirs(config.result_dir)
    
if not os.path.exists(config.checkpoint_dir):
    print("creating dir...", config.checkpoint_dir)
    os.makedirs(config.checkpoint_dir)

if not os.path.exists(config.result_img_dir):
    print("creating dir...", config.result_img_dir)
    os.makedirs(config.result_img_dir)
    
config.sample_dir = "samples_DIV2K"
if not os.path.exists(config.sample_dir):
    print("creating dir...", config.sample_dir)
    os.makedirs(config.sample_dir)

model = WESPE(config)
model.train()
model.test_generator(0, 11)
model.save()