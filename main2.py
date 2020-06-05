import os
import random
from data_loading import niftiDataGen
from model import unet3d, toy_unet3d

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

for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
    print(input_path, "|", target_path)


# Split our img paths into a training and a validation set
val_samples = 1
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]


# Instantiate data Sequences for each split
train_gen = niftiDataGen(batch_size, patch_size, extraction_step, train_input_img_paths, train_target_img_paths, num_classes)
val_gen = niftiDataGen(batch_size, patch_size, extraction_step, val_input_img_paths, val_target_img_paths, num_classes)


model = toy_unet3d(input_size = (*patch_size, 1), num_classes=num_classes)

# Configure the model for training.
# We use the "sparse" version of categorical_crossentropy
# because our target data is integers.

callbacks = [
    keras.callbacks.ModelCheckpoint("liver_segmentation.h5", save_best_only=True)
]

# Train the model, doing validation at the end of each epoch.
epochs = 15
model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)