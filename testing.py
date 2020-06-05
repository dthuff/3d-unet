import os
import random
from data_loading import niftiDataGen
from model import unet3d, toy_unet3d
import numpy as np

input_dir = "../data/image/"
target_dir = "../data/mask/"

patch_size = (64, 64, 64)
extraction_step = 64
num_classes = 2
batch_size = 1

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".nii.gz")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".nii.gz") and not fname.startswith(".")
    ]
)

print("Number of samples:", len(input_img_paths))

for input_path, target_path in zip(input_img_paths[:20], target_img_paths[:20]):
    print(input_path, "|", target_path)



# Instantiate data Sequences for each split
a_generator = niftiDataGen(batch_size=batch_size, 
                            patch_size=patch_size, 
                            extraction_step=extraction_step,
                            input_img_paths=input_img_paths,
                            target_img_paths=target_img_paths,
                            n_classes=num_classes)

for i in np.arange(10):
    data = a_generator.__getitem__(i)
    print(np.shape(data[0]))
    print(np.shape(data[1]))