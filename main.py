import numpy as np
from data_loading import load_nii
from data_loading import nifti_train_generator
from model import toy_unet3d



img_dir = './data/image/'
mask_dir = './data/mask/'
patchsize = (32, 32, 32) #(64, 64, 64) 
batchsize = 1
extractionstep = 64
seed = 1

m = toy_unet3d(input_size = (*patchsize, 1))

image_datagen = nifti_train_generator(img_dir=img_dir, 
                                    mask_dir=mask_dir, 
                                    batch_size=batchsize, 
                                    input_size=patchsize, 
                                    extraction_step=extractionstep)

m.fit_generator(image_datagen, steps_per_epoch=10, epochs=20, verbose=1)
