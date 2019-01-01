# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:54:22 2018
@author: SIMRAN TINANI
"""
'''
Adapted from the original code, URL: https://github.com/ctmakro/hellotensor/blob/master/lets_gan_clean.py

''' 
# Import the necessary packages

import tensorflow as tf
import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils import np_utils
import keras.backend as K
from keras.layers import concatenate
import math
import random
import numpy as np
import cv2


def dataset():
    # input image dimensions
    img_rows, img_cols = 32, 32
    
    img_channels = 3 # the flowers images are RGB, so we have 3 colour channels
   
    X_train = flowers
    
    print('X_train shape:', X_train.shape)
    print('Image Rows:', img_rows)
    print('Image Columns:', img_cols)
    print('Image Channels:', img_channels)

    X_train = X_train.astype('float32')
    
    X_train /= 255 # Normalize the pixel values to lie between 0 and 1 by dividing by 255

    X_train-=0.5 # Normalize the pixel values to lie between -0.5 and 0.5 by subtracting 0.5

    return X_train

xt = dataset()

inputdim=100

def generator(): # generative network architecture
    inp = Input(shape=(inputdim,))
    i = inp
    i = Reshape((1,1,inputdim))(i)

    ngf=24 # Number of frames in the generator

    def deconv(i,nop,kw,std=1,tail=True,bm='same'): # Transposed convolution 
        # Commonly, but controversially termed "deconvolution"
        global batch_size
        i = Conv2DTranspose(nop,kernel_size=(kw,kw),strides=(std,std),padding=bm)(i)
        if tail: # Batch Normalization and Relu are added to the deconv function itself
            i = bn(i)
            i = relu(i)
        return i

    i = deconv(i,nop=ngf*8,kw=4,std=1,bm='valid')
    i = deconv(i,nop=ngf*4,kw=4,std=2)
    i = deconv(i,nop=ngf*2,kw=4,std=2)
    i = deconv(i,nop=ngf*1,kw=4,std=2)

    i = deconv(i,nop=3,kw=4,std=1,tail=False) # out : 32x32x3
    i = Activation('tanh')(i)

    m = Model(inputs=inp,outputs=i)
    return m

def concat_diff(i): # batch discrimination -  increase generation diversity.
    # Avoiding mode collapse
    bv = Lambda(lambda x:K.mean(K.abs(x[:] - K.mean(x,axis=0)),axis=-1,keepdims=True))(i)
    # bv calculates the mean of the absolute differences between the batch images and 
    # their mean and concatenates it to the features extracted by the discriminator
    i= concatenate([i, bv], axis=-1)
    return i

def discriminator(): # Discriminative network architecture
    
    inp = Input(shape=(32,32,3))
    i = inp

    ndf=24 # number of frames in the discriminator

    def conv(i,nop,kw,std=1,usebn=True,bm='same'):
        i = Convolution2D(nop,(kw,kw),padding=bm,strides=(std,std))(i)
        if usebn:
            i = bn(i)
        i = relu(i)
        return i

    i = conv(i,ndf*1,4,std=2,usebn=False) # the first convolution does not use batch normalization
    i = concat_diff(i)
    i = conv(i,ndf*2,4,std=2)
    i = concat_diff(i)
    i = conv(i,ndf*4,4,std=2)
    i = concat_diff(i)
    i = conv(i,ndf*8,4,std=2)
    i = concat_diff(i)

    # 1x1
    i = Convolution2D(1,(2,2),padding='valid')(i)

    i = Activation('linear',name='conv_exit')(i)
    
    i = Activation('sigmoid')(i)

    i = Reshape((1,))(i) # predicted probability of image being real

    m = Model(inputs=inp,outputs=i)
    return m

print('generating G...')
gm = generator()
gm.summary()
print('generating D...')
dm = discriminator()
dm.summary()


def gan(g,d):
    # initialize a GAN trainer

    noise = Input(shape=g.input_shape[1:])
    real_data = Input(shape=d.input_shape[1:])

    generated = g(noise)
    gscore = d(generated)
    rscore = d(real_data)

    def log_eps(i):
        return K.log(i+1e-11)  # add a tiny number to avoid running into the singularity of log(0)

    dloss = - K.mean(log_eps(1-gscore) + .1 * log_eps(1-rscore) + .9 * log_eps(rscore))
    # loss of the generator, as first formulated by Ian Goodfellow, with single-side label smoothing
    gloss = - K.mean(log_eps(gscore))
    # loss of the generator, as first formulated by Ian Goodfellow
    Adam = tf.train.AdamOptimizer

    lr,b1 = 1e-4,.2 # lr is the learning rate. The same learning rate is used for the generator and discriminator
    optimizer = Adam(lr,beta1=b1)

    grad_loss_wd = optimizer.compute_gradients(dloss, d.trainable_weights)
    update_wd = optimizer.apply_gradients(grad_loss_wd)

    grad_loss_wg = optimizer.compute_gradients(gloss, g.trainable_weights)
    update_wg = optimizer.apply_gradients(grad_loss_wg)

    def get_internal_updates(model):
        # get all internal update ops (like moving averages) of a model
        inbound_nodes = model._inbound_nodes
        input_tensors = []
        for ibn in inbound_nodes:
            input_tensors+= ibn.input_tensors
        updates = [model.get_updates_for(i) for i in input_tensors]
        return updates

    other_parameter_updates = [get_internal_updates(m) for m in [d,g]]
    # those updates includes batch norm.

    print('other_parameter_updates for the models(mainly for batch norm):')
    print(other_parameter_updates)

    train_step = [update_wd, update_wg, other_parameter_updates]
    losses = [dloss,gloss]

    learning_phase = K.learning_phase()

    def gan_feed(sess,batch_image,z_input):
        # actual GAN trainer
        nonlocal train_step,losses,noise,real_data,learning_phase

        res = sess.run([train_step,losses],feed_dict={
        noise:z_input,
        real_data:batch_image,
        learning_phase:True,
        })

        loss_values = res[1]
        return loss_values #[dloss,gloss]

    return gan_feed
    
    
print('generating GAN...')
gan_feed = gan(gm,dm)

print('Ready. enter r() to train')

def r(ep=10000,noise_level=.01):
    sess = K.get_session()

    np.random.shuffle(xt)
    shuffled_cifar = xt
    length = len(shuffled_cifar)

    for i in range(ep):
        noise_level *= 0.99
        print('---------------------------')
        print('iter',i,'noise',noise_level)

        # sample from cifar
        j = i % int(length/batch_size)
        minibatch = shuffled_cifar[j*batch_size:(j+1)*batch_size]
        minibatch += np.random.normal(loc=0.,scale=noise_level,size=minibatch.shape)

        z_input = np.random.normal(loc=0.,scale=1.,size=(batch_size,zed))

        # train for one step
        losses = gan_feed(sess,minibatch,z_input)
        print('dloss:{:6.4f} gloss:{:6.4f}'.format(losses[0],losses[1]))

        if i==ep-1 or i % 10==0: show(save=True)

def autoscaler(img):
    limit = 400.
    # scales = [0.1,0.125,1./6.,0.2,0.25,1./3.,1./2.] + range(100)
    scales = np.hstack([1./np.linspace(10,2,num=9), np.linspace(1,100,num=100)])

    imgscale = limit/float(img.shape[0])
    for s in scales:
        if s>=imgscale:
            imgscale=s
            break

    img = cv2.resize(img,dsize=(int(img.shape[1]*imgscale),int(img.shape[0]*imgscale)),interpolation=cv2.INTER_NEAREST)

    return img,imgscale

def flatten_multiple_image_into_image(arr):
    import cv2
    num,uh,uw,depth = arr.shape

    patches = int(num+1)
    height = int(math.sqrt(patches)*0.9)
    width = int(patches/height+1)

    img = np.zeros((height*uh+height, width*uw+width, 3),dtype='float32')

    index = 0
    for row in range(height):
        for col in range(width):
            if index>=num-1:
                break
            channels = arr[index]
            img[row*uh+row:row*uh+uh+row,col*uw+col:col*uw+uw+col,:] = channels
            index+=1

    img,imgscale = autoscaler(img)

    return img,imgscale

def show(save=False):
    i = np.random.normal(loc=0.,scale=1.,size=(batch_size,zed))
    gened = gm.predict([i])

    gened *= 0.5
    gened +=0.5

    im,ims = flatten_multiple_image_into_image(gened)
    cv2.imshow('gened scale:'+str(ims),im)
    cv2.waitKey(1)

    if save!=False:
        cv2.imwrite("image"+str(i)+".jpg",im*255)    
        
r()

# Saving the results 

cv2.destroyAllWindows() # first destroy the open cv2 window
noise = np.random.normal(loc=0.,scale=1.,size=(100,inputdim))
gened = gm.predict([noise])
b=dm.predict(xt) # predicted probabilities on the real data: the closer these are to 0.5, the better the convergence
gened *= 0.5
gened +=0.5
im,ims = flatten_multiple_image_into_image(gened)

cv2.imwrite("100flowers.jpg",im*255) # saves the image in the WD


