import os
import random
from data_loading import niftiDataGen, load_nii
import numpy as np

input_dir = "../data/multilabel/image/"
target_dir = "../data/multilabel/mask/"

patch_size = (32, 32, 32)
extraction_step = 16
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

for input_path, target_path in zip(input_img_paths, target_img_paths):
    print(input_path, "|", target_path)
    img = load_nii(input_path)
    print(np.shape(img))
    mask = load_nii(target_path)
    print(np.shape(mask))
    if not np.shape(img) == np.shape(mask):
        print("****************UH OH****************")
    



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