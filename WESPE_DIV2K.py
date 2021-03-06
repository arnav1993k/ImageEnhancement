from __future__ import division
import os
import time
import tensorflow as tf
import scipy.misc
import scipy.io
import numpy as np
from glob import glob
from utils import *
from ops import *
from vgg19 import *
from dataloader.dataloader_DIV2K import *
import modules
from math import sqrt as sqrt

class WESPE(object):
    def __init__(self, sess, config, dataset_phone, dataset_canon, dataset_DIV2K):
        # copy training parameters
        self.sess = sess
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
        
        self.dataset_name = config.dataset_name
        self.dataset_phone = dataset_phone
        self.dataset_canon = dataset_canon
        self.dataset_DIV2K = dataset_DIV2K
        
        # loss weights
        self.w_content = config.w_content
        self.w_profile = config.w_profile
        self.w_texture = config.w_texture 
        self.w_color = config.w_color
        self.w_tv = config.w_tv
        self.gamma = config.gamma

        #total losses
        self.total_profile_loss = 0
        self.total_color_loss = 0
        self.total_var_loss = 0
        self.total_texture_loss = 0
        self.total_content_loss = 0

        # patches for training (fixed size)
        self.phone_patch = tf.placeholder(tf.float32, [self.batch_size, self.patch_size, self.patch_size, self.channels], name='input_phone_patch')
        self.canon_patch = tf.placeholder(tf.float32, [self.batch_size, self.patch_size, self.patch_size, self.channels], name='input_canon_patch')
        self.DIV2K_patch = tf.placeholder(tf.float32, [self.batch_size, self.patch_size, self.patch_size, self.channels], name='input_DIV2K_patch') 
        
        # images for testing (unknown size)
        self.phone_test = tf.placeholder(tf.float32, [None, self.patch_size, self.patch_size, self.channels], name='input_phone_test')
        self.phone_test_unknown = tf.placeholder(tf.float32, [None, None, None, self.channels], name='input_phone_test_unknown_size')
        
        # input to discriminator network
        self.input_discriminator = tf.placeholder(tf.float32, [self.batch_size, self.patch_size, self.patch_size, self.channels], name='input_discriminator') 
        
        # builc models
        self.build_generator()
        self.build_discriminator()
        
        # build loss function (color + texture + content + TV)
        self.build_generator_loss()
        tf.global_variables_initializer().run(session=self.sess)
        
        self.saver = tf.train.Saver(tf.trainable_variables())


    def build_generator(self):
        self.enhanced_patch = modules.generator_network(self.phone_patch, var_scope = 'generator')
        #self.reconstructed_patch = modules.generator_network(self.enhanced_patch, var_scope = 'generator_inverse')
        
        self.enhanced_test = modules.generator_network(self.phone_test, var_scope = 'generator')
        #self.reconstructed_test = modules.generator_network(self.enhanced_test, var_scope = 'generator_inverse')
        self.enhanced_test_unknown = modules.generator_network(self.phone_test_unknown, var_scope = 'generator')
        #self.reconstructed_test_unknown = modules.generator_network(self.enhanced_test_unknown, var_scope = 'generator_inverse')
        
        variables = tf.trainable_variables()
        self.g_var = [x for x in variables if 'generator' in x.name]
        print("Completed building generator. Number of variables:",len(self.g_var))
        #print(self.g_var)

    def build_generator_loss(self):
        # content loss (vgg feature distance between original & reconstructed)
        original_vgg = net(self.vgg_dir, self.canon_patch * 255)
        enhanced_vgg = net(self.vgg_dir, self.enhanced_patch * 255)
        
        #content loss
        content_loss = tf.reduce_mean(tf.square(original_vgg[self.content_layer] - enhanced_vgg[self.content_layer])) 
        
        #profile loss(gan, enhanced-div2k)
        profile_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.logits_DIV2K_profile, self.logits_enhanced_profile))
        
        # color loss (gan, enhanced-original)
        color_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.logits_original_color, self.logits_enhanced_color))
        
        # texture loss (gan, enhanced-original)
        texture_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.logits_original_texture, self.logits_enhanced_texture))
        
        # tv loss (total variation of enhanced)
        tv_loss = tf.reduce_mean(tf.abs(tf.image.total_variation(self.enhanced_patch)-tf.image.total_variation(self.canon_patch)))
        
        #computing expected mean sq.loss
#         self.total_content_loss = self.gamma*self.total_content_loss+(1-self.gamma)*tf.square(self.content_loss)
#         self.total_var_loss = self.gamma*self.total_var_loss+(1-self.gamma)*tf.square(self.tv_loss)
#         self.total_color_loss = self.gamma*self.total_color_loss+(1-self.gamma)*tf.square(self.color_loss)
#         self.total_texture_loss = self.gamma*self.total_texture_loss+(1-self.gamma)*tf.square(self.texture_loss)

        # calculate generator loss as a weighted sum of the above 4 losses
#         self.G_loss = (1-self.gamma)*(tf.square(self.color_loss) * self.w_color/tf.sqrt(self.total_color_loss) + tf.square(self.texture_loss) * self.w_texture/tf.sqrt(self.total_texture_loss) + tf.square(self.content_loss) * self.w_content/tf.sqrt(self.total_content_loss) + tf.square(self.tv_loss) * self.w_tv/tf.sqrt(self.total_var_loss))
        self.content_loss = content_loss
        self.tv_loss = tv_loss
        self.color_loss = color_loss
        self.profile_loss = profile_loss
        self.texture_loss = texture_loss
        self.G_loss = self.w_content*content_loss+self.w_profile*profile_loss+self.w_color*color_loss+self.w_texture*texture_loss+self.w_tv*tv_loss
        self.G_optimizer = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.G_loss, var_list=self.g_var)
    
    def build_discriminator_unit(self, generated_patch, actual_patch, preprocess, scope_name):
        actual, _ = modules.discriminator_network(actual_patch, var_scope = scope_name, preprocess = preprocess)
        fake ,_ = modules.discriminator_network(generated_patch, var_scope = scope_name, preprocess = preprocess)
        variables = tf.trainable_variables()
        d_vars = [x for x in variables if scope_name in x.name]
        print("Completed building %s discriminator. Number of variables = %d" %(scope_name,len(d_vars)))

        loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(actual, tf.ones_like(actual)))
        loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(fake, tf.zeros_like(fake)))
        total_loss = loss_real+loss_fake

        return total_loss, d_vars, actual, fake

    def build_discriminator(self):
        self.d_loss_profile, self.d_var_profile, self.logits_DIV2K_profile, self.logits_enhanced_profile = self.build_discriminator_unit(self.enhanced_patch,self.DIV2K_patch,'blur','discriminator_profile')
        self.d_loss_color, self.d_var_color, self.logits_original_color, self.logits_enhanced_color = self.build_discriminator_unit(self.enhanced_patch,self.canon_patch,'none','discriminator_color')
        self.d_loss_texture, self.d_var_texture, self.logits_original_texture, self.logits_enhanced_texture = self.build_discriminator_unit(self.enhanced_patch,self.canon_patch,'gray','discriminator_texture')

        #optimizers
        self.D_optimizer_profile = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.d_loss_profile, var_list=self.d_var_profile)
        self.D_optimizer_color = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.d_loss_color, var_list=self.d_var_color)
        self.D_optimizer_texture = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.d_loss_texture, var_list=self.d_var_texture)

#     def build_discriminator(self): 
#         self.logits_DIV2K_color, _ = modules.discriminator_network(self.DIV2K_patch, var_scope = 'discriminator_color', preprocess = 'blur')
# #         self.logits_DIV2K_color, _ = modules.discriminator_network(self.DIV2K_patch, var_scope = 'discriminator_color', preprocess = 'none')
#         self.logits_DIV2K_texture, _ = modules.discriminator_network(self.canon_patch, var_scope = 'discriminator_texture', preprocess = 'gray')
        
#         self.logits_enhanced_color, _ = modules.discriminator_network(self.enhanced_patch, var_scope = 'discriminator_color', preprocess = 'blur')
# #         self.logits_enhanced_color, _ = modules.discriminator_network(self.enhanced_patch, var_scope = 'discriminator_color', preprocess = 'none')
#         self.logits_enhanced_texture, _ = modules.discriminator_network(self.enhanced_patch, var_scope = 'discriminator_texture', preprocess = 'gray')
        
#         #_, self.prob = modules.discriminator_network(self.phone_test)
           
#         variables = tf.trainable_variables()
#         self.d_var_color = [x for x in variables if 'discriminator_color' in x.name]
#         print("Completed building color discriminator. Number of variables:",len(self.d_var_color))
        
#         d_loss_real_color = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.logits_DIV2K_color, tf.ones_like(self.logits_DIV2K_color)))
#         d_loss_fake_color = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.logits_enhanced_color, tf.zeros_like(self.logits_enhanced_color)))
        
#         self.d_loss_color = d_loss_real_color + d_loss_fake_color
#         self.D_optimizer_color = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.d_loss_color, var_list=self.d_var_color)
        
#         self.d_var_texture = [x for x in variables if 'discriminator_texture' in x.name]
#         print("Completed building texture discriminator. Number of variables:",len(self.d_var_texture))
        
#         d_loss_real_texture = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.logits_DIV2K_texture, tf.ones_like(self.logits_DIV2K_texture)))
#         d_loss_fake_texture = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.logits_enhanced_texture, tf.zeros_like(self.logits_enhanced_texture)))
        
#         self.d_loss_texture = d_loss_real_texture + d_loss_fake_texture
#         self.D_optimizer_texture = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.d_loss_texture, var_list=self.d_var_texture)
        
    def train(self, load = True):
        if load == True:
            if self.load():
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        else:
            print(" Overall training starts from beginning")
        start = time.time()
        for i in range(0, self.config.train_iter):
            #create batches
            phone_batch, canon_batch, DIV2K_batch = get_batch(self.dataset_phone, self.dataset_canon, self.dataset_DIV2K, self.config)
            
            #Run generator
            _, enhanced_batch = self.sess.run([self.G_optimizer, self.enhanced_patch] , feed_dict={self.phone_patch:phone_batch, self.canon_patch:canon_batch, self.DIV2K_patch:DIV2K_batch})
            
            #Run discriminator for profile
            _ = self.sess.run(self.D_optimizer_profile , feed_dict={self.phone_patch:phone_batch, self.canon_patch:canon_batch, self.DIV2K_patch:DIV2K_batch})
            
            #Run discriminator for color
            _ = self.sess.run(self.D_optimizer_color , feed_dict={self.phone_patch:phone_batch, self.canon_patch:canon_batch, self.DIV2K_patch:DIV2K_batch})

            #Run discriminator for texture
            _ = self.sess.run(self.D_optimizer_texture , feed_dict={self.phone_patch:phone_batch, self.canon_patch:canon_batch, self.DIV2K_patch:DIV2K_batch})
            
            if i %self.config.test_every == 0:
                phone_batch, canon_batch, DIV2K_batch = get_batch(self.dataset_phone, self.dataset_canon, self.dataset_DIV2K, self.config)
                print(phone_batch.shape,canon_batch.shape,DIV2K_batch.shape)
                g_loss, content_loss, profile_loss, color_loss, texture_loss, tv_loss = self.sess.run([self.G_loss, self.content_loss, self.profile_loss, self.color_loss, self.texture_loss, self.tv_loss] , feed_dict={self.phone_patch:phone_batch, self.canon_patch:canon_batch, self.DIV2K_patch:DIV2K_batch})
                print("Iteration %d, runtime: %.3f s, generator loss: %.6f" %(i, time.time()-start, g_loss))      
                print("Loss per component: content %.6f,profile %.6f, color %.6f, texture %.6f, tv %.6f" %(content_loss, profile_loss, color_loss, texture_loss, tv_loss))
                #print("Loss per component: total_content %.4f, total_color %.4f, total_texture %.4f, total_tv %.4f" %(sqrt(total_content_loss), sqrt(total_color_loss), sqrt(total_texture_loss), sqrt(total_var_loss)))
                # during training, test for only patches (full image testing incurs memory issues...)
                self.test_generator(100, 0)
                self.save()
    
    def test_generator(self, test_num_patch = 200, test_num_image = 5, load = False):
        if load == True:
            if self.load():
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        # test for patches
        start = time.time()
        test_list_phone = sorted(glob(self.config.test_path_phone_patch))

        #PSNR_phone_reconstructed_list = np.zeros([test_num_patch])
        PSNR_phone_enhanced_list = np.zeros([test_num_patch])

        indexes = []
        for i in range(test_num_patch):
            index = np.random.randint(len(test_list_phone))
            indexes.append(index)
            test_img = scipy.ndimage.imread(test_list_phone[index], mode = "RGB").astype("float32")
            test_patch_phone = get_patch(test_img,self.config.patch_size)
            test_patch_phone = preprocess(test_patch_phone)

            test_patch_enhanced= self.sess.run([self.enhanced_test] , feed_dict={self.phone_test:[test_patch_phone]})
            if i % 50 == 0:

                imageio.imwrite(("%s/phone_%d.png" %(self.result_img_dir, i)), postprocess(test_patch_phone))
                imageio.imwrite(("%s/enhanced_%d.png" %(self.result_img_dir,i)), postprocess(test_patch_enhanced[0][0]))

            PSNR = calc_PSNR(postprocess(test_patch_enhanced[0]), postprocess(test_patch_phone))
            #print("PSNR: %.3f" %PSNR)
            PSNR_phone_enhanced_list[i] = PSNR

            #PSNR = calc_PSNR(postprocess(test_patch_reconstructed[0]), postprocess(test_patch_phone))
            #print("PSNR: %.3f" %PSNR)
            #PSNR_phone_reconstructed_list[i] = PSNR
        print("(runtime: %.3f s) Average test PSNR for %d random test image patches: phone-enhanced %.3f" %(time.time()-start, test_num_patch, np.mean(PSNR_phone_enhanced_list)))
        
        # test for images
        start = time.time()
        test_list_phone = sorted(glob(self.config.test_path_phone_image))
        PSNR_phone_enhanced_list = np.zeros([test_num_image])
        #PSNR_phone_reconstructed_list = np.zeros([test_num_image])

        indexes = []
        for i in range(test_num_image):
            #index = np.random.randint(len(test_list_phone))
            index = i
            indexes.append(index)
            test_image_phone = preprocess(scipy.misc.imread(test_list_phone[index], mode = "RGB").astype("float32"))

            test_image_enhanced = self.sess.run([self.enhanced_test_unknown] , feed_dict={self.phone_test_unknown:[test_image_phone]})
            imageio.imwrite(("%s/phone_%d.png" %(self.sample_dir, i)), postprocess(test_image_phone))
            imageio.imwrite(("%s/enhanced_%d.png" %(self.sample_dir, i)), postprocess(test_image_enhanced[0][0]))
            #imageio.imwrite(("./samples_DIV2K/%s/image/reconstructed_%d.png" %(self.config.dataset_name, i)), postprocess(test_image_reconstructed[0]))
            
            PSNR = calc_PSNR(postprocess(test_image_enhanced[0]), postprocess(test_image_phone))
            #print("PSNR: %.3f" %PSNR)
            PSNR_phone_enhanced_list[i] = PSNR
            
            #PSNR = calc_PSNR(postprocess(test_image_reconstructed[0]), postprocess(test_image_phone))
            #PSNR_phone_reconstructed_list[i] = PSNR
        if test_num_image > 0:
            print("(runtime: %.3f s) Average test PSNR for %d random full test images: original-enhanced %.3f" %(time.time()-start, test_num_image, np.mean(PSNR_phone_enhanced_list)))

    def save(self):
        model_name = self.config.model_name
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.dataset_name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), write_meta_graph=False)

    def load(self):
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.dataset_name)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        print("Loading checkpoints from ",checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            model_name = self.config.model_name
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, model_name))
            return True
        else:
            return False