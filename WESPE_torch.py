import time
from glob import glob
import numpy as np
import scipy.misc
import scipy.io
import imageio
import torch
import torch.nn as nn

from ImageEnhancement.generator import Generator
from ImageEnhancement.discriminator import Discriminator
from ImageEnhancement.dataloader.dataloader_torch import Dataset, get_patch
from ImageEnhancement.vgg19_torch import net
from ImageEnhancement.ops_torch import preprocess, postprocess
from ImageEnhancement.utils.utils import calc_PSNR



class WESPE(object):
    """docstring for WESPE"""
    def __init__(self, config):
        super(WESPE, self).__init__()
        
        self.config = config
        self.batch_size = config.batch_size
        self.patch_size = config.patch_size
        self.mode = config.mode
        self.channels = config.channels
        self.augmentation = config.augmentation
        self.checkpoint_dir = config.checkpoint_dir
        self.sample_dir = config.sample_dir
        self.result_img_dir = config.result_img_dir
        self.content_layer = config.content_layer
        self.vgg_dir = config.vgg_dir

        # Data
        self.dataset_name = config.dataset_name
        self.dataset = Dataset(self.config)
        self.data_loader = torch.utils.data.DataLoader(self.dataset,
                                                       batch_size=self.config.batch_size,
                                                       shuffle=True,
                                                       num_workers=self.config.data_loader_workers,
                                                       pin_memory=self.config.pin_memory,
                                                       drop_last=True)

        # Loss Weights
        self.w_content = config.w_content
        self.w_profile = config.w_profile
        self.w_texture = config.w_texture 
        self.w_color = config.w_color
        self.w_tv = config.w_tv
        self.gamma = config.gamma

        # Total Losses
        self.total_profile_loss = 0
        self.total_color_loss = 0
        self.total_var_loss = 0
        self.total_texture_loss = 0
        self.total_content_loss = 0

        # Networks
        self.generator = Generator()
        self.discriminator1 = Discriminator(in_channels=3)
        self.discriminator2 = Discriminator(in_channels=3)
        self.discriminator3 = Discriminator(in_channels=1)
        
        if torch.cuda.is_available():
            self.generator, self.discriminator1, self.discriminator2, self.discriminator3 = \
            self.generator.cuda(), self.discriminator1.cuda(), self.discriminator2.cuda(), self.discriminator3.cuda()

        # Network Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters())
        self.optimizer_D1 = torch.optim.Adam(self.discriminator1.parameters())
        self.optimizer_D2 = torch.optim.Adam(self.discriminator2.parameters())
        self.optimizer_D3 = torch.optim.Adam(self.discriminator3.parameters())

        # Discriminator Loss Function
        self.loss_fn_D = nn.BCEWithLogitsLoss()
        

    def build_discriminator_unit(self, generated_patch, actual_batch, index, preprocess):

        if index == 1:
            act, _ = self.discriminator1(actual_batch, preprocess = preprocess)
            fake, _ = self.discriminator1(generated_patch, preprocess = preprocess)

        elif index == 2:
            act, _ = self.discriminator2(actual_batch, preprocess = preprocess)
            fake, _ = self.discriminator2(generated_patch, preprocess = preprocess)

        elif index == 3:
            act, _ = self.discriminator3(actual_batch, preprocess = preprocess)
            fake, _ = self.discriminator3(generated_patch, preprocess = preprocess)

        else:
            raise NotImplementedError

        loss_real = self.loss_fn_D(act, torch.ones_like(act))
        loss_fake = self.loss_fn_D(fake, torch.zeros_like(fake))
        total_loss = loss_real+loss_fake

        return total_loss, act, fake
    
    
    
    def train(self):
        
        for i in range(self.config.train_iter):
            self.train_one_epoch(i)

            
    def train_one_epoch(self):
        
        start = time.time()
        
        self.total_profile_loss = 0
        self.total_color_loss = 0
        self.total_var_loss = 0
        self.total_texture_loss = 0
        self.total_content_loss = 0

        for i in range(self.config.train_iter):

            for step, (phone_patch, canon_patch, DIV2K_patch) in enumerate(self.data_loader):

                phone_patch, canon_patch, DIV2K_patch = phone_patch.float(), canon_patch.float(), DIV2K_patch.float()
                
                if torch.cuda.is_available():
                    phone_patch, canon_patch, DIV2K_patch = phone_patch.cuda(), canon_patch.cuda(), DIV2K_patch.cuda()
                
                self.optimizer_G.zero_grad()

                # Generator
                enhanced_patch = self.generator(phone_patch)
                
                # Discrimiator 1
                d_loss_profile, logits_DIV2K_profile, logits_enhanced_profile = self.build_discriminator_unit(enhanced_patch, DIV2K_patch, index=1, preprocess='blur')
                
                # Discrimiator 2
                d_loss_color, logits_original_color, logits_enhanced_color = self.build_discriminator_unit(enhanced_patch, canon_patch, index=2, preprocess='none')
                
                # Discrimiator 3
                d_loss_texture, logits_original_texture, logits_enhanced_texture = self.build_discriminator_unit(enhanced_patch, canon_patch, index=3, preprocess='gray')

                # Generator Loss
                original_vgg = net(self.vgg_dir, canon_patch * 255)
                enhanced_vgg = net(self.vgg_dir, enhanced_patch * 255)
                
                #content loss
                content_loss = torch.mean(torch.pow(original_vgg[self.content_layer] - enhanced_vgg[self.content_layer], 2))
                
                #profile loss(gan, enhanced-div2k)
                profile_loss = self.loss_fn_D(logits_DIV2K_profile, logits_enhanced_profile)
                
                # color loss (gan, enhanced-original)
                color_loss = self.loss_fn_D(logits_original_color, logits_enhanced_color)
                
                # texture loss (gan, enhanced-original)
                texture_loss = self.loss_fn_D(logits_original_texture, logits_enhanced_texture)
                
                # tv loss (total variation of enhanced)
                tv_loss = torch.mean(torch.abs(self.total_variation_loss(enhanced_patch) - self.total_variation_loss(canon_patch)))

                g_loss = self.w_content*content_loss + self.w_profile*profile_loss + self.w_color*color_loss + self.w_texture*texture_loss + self.w_tv*tv_loss

                g_loss.backward(retain_graph=True)
                self.optimizer_G.step()

                self.optimizer_D1.zero_grad()
                self.optimizer_D2.zero_grad()
                self.optimizer_D3.zero_grad()

                d_loss_profile.backward()
                self.optimizer_D1.step()

                d_loss_color.backward()
                self.optimizer_D2.step()

                d_loss_texture.backward()
                self.optimizer_D3.step()
                
                self.total_profile_loss += profile_loss
                self.total_color_loss += color_loss
                self.total_var_loss += tv_loss
                self.total_texture_loss += texture_loss
                self.total_content_loss += content_loss
                
                if i %self.config.test_every == 0:
                    print("Iteration %d, runtime: %.3f s, generator loss: %.6f" %(i, time.time() - start, g_loss))      
                    print("Loss per component: content %.6f,profile %.6f, color %.6f, texture %.6f, tv %.6f" %(content_loss, profile_loss, color_loss, texture_loss, tv_loss))
                    self.test_generator(100, 0)
    



    def test_generator(self, test_num_patch = 200, test_num_image = 5, load = False):
        
        self.generator.eval()
        
        # test for patches
        start = time.time()
        test_list_phone = sorted(glob(self.config.test_path_phone_patch))
        PSNR_phone_enhanced_list = np.zeros([test_num_patch])
        
        indexes = []
        for i in range(test_num_patch):
            index = np.random.randint(len(test_list_phone))
            indexes.append(index)
            test_img = scipy.misc.imread(test_list_phone[index], mode = "RGB").astype("float32")
            test_patch_phone = get_patch(test_img, self.config.patch_size)
            test_patch_phone = preprocess(test_patch_phone)
            
            with torch.no_grad():
                test_patch_phone = torch.from_numpy(np.transpose(test_patch_phone, (2,1,0))).float().unsqueeze(0)
                if torch.cuda.is_available():
                    test_patch_phone = test_patch_phone.cuda()

                test_patch_enhanced = self.generator(test_patch_phone)
            
            test_patch_enhanced = test_patch_enhanced.cpu().data.numpy()
            test_patch_enhanced = np.transpose(test_patch_enhanced.cpu().data.numpy(), (0,2,3,1))
            test_patch_phone = np.transpose(test_patch_phone.cpu().data.numpy(), (0,2,3,1))

            if i % 50 == 0:
                imageio.imwrite(("%s/phone_%d.png" %(self.result_img_dir, i)), postprocess(test_patch_phone[0]))
                imageio.imwrite(("%s/enhanced_%d.png" %(self.result_img_dir,i)), postprocess(test_patch_enhanced[0]))

            PSNR = calc_PSNR(postprocess(test_patch_enhanced[0]), postprocess(test_patch_phone))
            PSNR_phone_enhanced_list[i] = PSNR

        print("(runtime: %.3f s) Average test PSNR for %d random test image patches: phone-enhanced %.3f" %(time.time()-start, test_num_patch, np.mean(PSNR_phone_enhanced_list)))
        
        # test for images
        start = time.time()
        test_list_phone = sorted(glob(self.config.test_path_phone_image))
        PSNR_phone_enhanced_list = np.zeros([test_num_image])

        indexes = []
        for i in range(test_num_image):
            index = i
            indexes.append(index)
            
            test_image_phone = preprocess(scipy.misc.imread(test_list_phone[index], mode = "RGB").astype("float32"))
            
            with torch.no_grad():
                test_image_phone = torch.from_numpy(np.transpose(test_image_phone, (2,1,0))).float().unsqueeze(0)
                if torch.cuda.is_available():
                    test_image_phone = test_image_phone.cuda()

                test_image_enhanced = self.generator(test_image_phone)
            
            test_image_enhanced = test_image_enhanced.cpu().data.numpy()
            test_image_enhanced = np.transpose(test_image_enhanced.cpu().data.numpy(), (0,2,3,1))
            test_image_phone = np.transpose(test_image_phone.cpu().data.numpy(), (0,2,3,1))
                        
            imageio.imwrite(("%s/phone_%d.png" %(self.sample_dir, i)), postprocess(test_image_phone[0]))
            imageio.imwrite(("%s/enhanced_%d.png" %(self.sample_dir, i)), postprocess(test_image_enhanced[0]))
            
            PSNR = calc_PSNR(postprocess(test_image_enhanced[0]), postprocess(test_image_phone[0]))
            PSNR_phone_enhanced_list[i] = PSNR
            
        if test_num_image > 0:
            print("(runtime: %.3f s) Average test PSNR for %d random full test images: original-enhanced %.3f" %(time.time()-start, test_num_image, np.mean(PSNR_phone_enhanced_list)))


    def total_variation_loss(self, images):

        ndims = len(images.shape)

        if ndims == 3:
            pixel_dif1 = images[:, 1:, :] - images[:, :-1, :]
            pixel_dif2 = images[:, :, 1:] - images[:, :, :-1]
            sum_axis = None

        if ndims == 4:
            pixel_dif1 = images[:, :, 1:, :] - images[:, :, :-1, :]
            pixel_dif2 = images[:, :, :, 1:] - images[:, :, :, :-1]
            sum_axis = (1, 2, 3)

        else:
            raise ValueError('\'images\' must be either 3 or 4-dimensional.')

        tot_var = (
            torch.sum(torch.abs(pixel_dif1)) +
            torch.sum(torch.abs(pixel_dif2), dim=sum_axis))

        return tot_var
                
            
    