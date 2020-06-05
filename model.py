import numpy as np 
import os
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from loss import tversky_loss, dice_coef_ignore_bg, dice_coef_ignore_bg_loss


def unet3d(pretrained_weights = None, input_size = (64,64,64,1), num_classes=19):

    inputs = Input(input_size)
    conv1 = Conv3D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv3D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)

    up5 = Conv3D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(drop4))
    merge5 = concatenate([conv3,up5], axis = 4)
    conv5 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge5)
    conv5 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

    up6 = Conv3D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(conv5))
    merge6 = concatenate([conv2,up6], axis = 4)
    conv6 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv3D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(conv6))
    merge7 = concatenate([conv1,up7], axis = 4)
    conv7 = Conv3D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv3D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    outputs = Conv3D(num_classes, 1, activation = 'sigmoid')(conv7)

    model = Model(inputs = inputs, outputs = outputs)

    model.compile(optimizer = Adam(lr = 1e-4), loss = dice_coef_ignore_bg_loss, metrics = [tf.keras.metrics.MeanIoU(num_classes=2), dice_coef_ignore_bg])
    
    model.summary(line_length=110)

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


def toy_unet3d(pretrained_weights = None, input_size = (64,64,64,1), num_classes=19):

    inputs = Input(input_size)
    conv1 = Conv3D(10, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv10 = Conv3D(num_classes, 1, activation = 'sigmoid')(conv1)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = tversky_loss, metrics = ['accuracy'])
    
    model.summary(line_length=110)

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
