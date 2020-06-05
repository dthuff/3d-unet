import numpy as np
from data_loading import load_nii
from data_loading import nifti_train_generator
from model import unet3d
from keras.callbacks import ModelCheckpoint

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use id from $ nvidia-smi

img_dir = '../data/image/'
mask_dir = '../data/mask/'
patchsize = (32, 32, 32) #(64, 64, 64) 
batchsize = 1
extractionstep = 64

m = unet3d(input_size = (*patchsize, 1), num_classes=2)

image_datagen = nifti_train_generator(img_dir=img_dir, 
                                    mask_dir=mask_dir, 
                                    batch_size=batchsize, 
                                    input_size=patchsize, 
                                    extraction_step=extractionstep)

epochs=10
steps_per_epoch = 20
callbacks = [ModelCheckpoint("liver_segmentation.h5", save_best_only=True)]

m.fit_generator(image_datagen, steps_per_epoch=steps_per_epoch,  epochs=epochs, callbacks=callbacks)
#m.fit_generator(image_datagen, steps_per_epoch=10, epochs=20, verbose=1)

