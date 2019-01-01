# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:54:22 2018

@author: SIMRAN TINANI
"""
'''
Adapted from the original code: DCGAN on MNIST using Keras, Author: Rowel Atienza 
URL: https://github.com/roatienza/Deep-Learning-Experiments
'''


from keras import backend as K
K.tensorflow_backend._get_available_gpus()
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import random
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )


class DCGAN(object):
    def __init__(self, img_rows=28, img_cols=28, channel=1):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    def discriminator(self):
        if self.D: # if the discriminator has been provided by the user, return/use that 
            return self.D
        self.D = Sequential()
        depth = 64
        dropout = 0.4
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape, padding='same')) # 5*5 filters
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))
        self.D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))
        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))
        self.D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))
        
        # 5*5 filters

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1)) # 1 output neuron
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential()
        dropout = 0.4
        depth = 64+64+64+64
        dim = 7
        # In: 100
        # Out: dim x dim x depth
        self.G.add(Dense(dim*dim*depth, input_dim=100)) # First: A Dense Layer
        self.G.add(BatchNormalization(momentum=0.5)) # Batch Normalization
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim, dim, depth)))
        self.G.add(Dropout(dropout))

        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same')) # Deconvolution/transposed convolution
        self.G.add(BatchNormalization(momentum=0.5))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.5))
        self.G.add(Activation('relu'))

        self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.5))
        self.G.add(Activation('relu'))

        # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
        self.G.add(Conv2DTranspose(1, 5, padding='same'))
        self.G.add(Activation('sigmoid'))
        self.G.summary()
        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.00015, decay=4e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.00005, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator()) # add the generative architecture
        self.discriminator().trainable = False #### REQUIRED
        self.AM.add(self.discriminator()) # add the discriminative architecture
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy']) # output of the generator is fed as input of the discriminator
        return self.AM # accuracy returned is the accuracy of discriminator on generator images

class flowers_DCGAN(object):
    def __init__(self): # initializing MNIST_DCGAN
        self.img_rows = 28
        self.img_cols = 28
        self.channel = 1
        self.x_train = flowersgs
        self.x_train = self.x_train.reshape(-1, self.img_rows,\
        	self.img_cols, 1).astype(np.float32) # reshape 784*1 to 28*28*1

        self.DCGAN = DCGAN()
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    def train(self, train_steps=2000, batch_size=28, save_interval=0): # batch size for minibatch GD
        noise_input = None
        if save_interval>0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100]) # input noise (z-value to generator)
        for i in range(train_steps):
            images_train = self.x_train[np.random.randint(0, # pick batch_size number of random images from the training set
                self.x_train.shape[0], size=batch_size), :, :, :]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100]) # input noise (z-value to generator)
            images_fake = self.generator.predict(noise) # Generates output predictions for the input samples.
# generates output images using the noise
            x = np.concatenate((images_train, images_fake)) # real and fake images
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0 # y contains the labels: 1 for real and 0 for fake images
            d_loss = self.discriminator.train_on_batch(x, y) # Runs a single gradient update on a single batch of data.
            # Discriminator is trained on both real and fake images
            # Loss of discriminator calculated. Now re-define labels and noise
            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100]) # 100 is the input size to generator
            a_loss = self.adversarial.train_on_batch(noise, y) # feed in noise, generator comes first in the adversarial model
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1]) # display msgs
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1]) # display msgs
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0],\
                        noise=noise_input, step=(i+1))

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'flowersgs.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "flowersgs_%d.png" % step
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()
            
if __name__ == '__main__':
    flowers_dcgan = flowers_DCGAN()
    timer = ElapsedTimer()
    flowers_dcgan.train(train_steps=10000, batch_size=28, save_interval=1000)
    timer.elapsed_time()
    flowers_dcgan.plot_images(fake=True)
    flowers_dcgan.plot_images(fake=False, save2file=True)
