import os
import random
import numpy as np
from data_loading import niftiDataGen
from model import unet3d_3blocks, toy_unet3d
from loss import *
from keras.optimizers import *
from keras.callbacks import *
from matplotlib import pyplot

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use id from $ nvidia-smi

input_dir = "../data/multilabel/image/"
target_dir = "../data/multilabel/mask/"

patch_size = (16, 16, 16)
extraction_step = 16
num_classes = 6 #including background (e.g. num_classes must be >=2)
batch_size = 1
val_fraction = 0.2
epochs = 200

# Sort and display available data
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

for input_path, target_path in zip(input_img_paths[:93], target_img_paths[:93]):
    print(input_path, "|", target_path)


# Split our img paths into a training and a validation set
val_samples = np.floor(val_fraction * len(input_img_paths)).astype(int)
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]


# Instantiate data Sequences for each split
train_gen = niftiDataGen(batch_size, 
                        patch_size, 
                        extraction_step, 
                        train_input_img_paths, 
                        train_target_img_paths, 
                        num_classes)

val_gen = niftiDataGen(batch_size, 
                        patch_size, 
                        extraction_step, 
                        val_input_img_paths, 
                        val_target_img_paths, 
                        num_classes)

# Instantiate and compile the model
model = unet3d_3blocks(input_size = (*patch_size, 1), num_classes=num_classes)

model.compile(optimizer = Adam(lr = 1e-4), 
            loss = tversky_loss, 
            metrics =  [dice_for_class(i) for i in range(num_classes)])

#[dice_for_class(0), dice_for_class(1), dice_coef_ignore_bg])

# Instantiate callbacks
#    ModelCheckpoint(filepath='./saved_models/model.{epoch:02d}-{val_loss:.2f}.h5'),

callbacks = [
    TensorBoard(log_dir="./logs"),
    ModelCheckpoint(filepath='./saved_models/liver_segmentation.h5', save_best_only=True),
    EarlyStopping(monitor='loss', min_delta=0.002, patience=5, verbose=0, restore_best_weights=True)
]

# Train the model, doing validation at the end of each epoch.
history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)

#Plot training curve
pyplot.plot(history.history['dice_inner'])
pyplot.show()



trained_model = unet3d_3blocks(pretrained_weights='./saved_models/liver_segmentation.h5',input_size = (*patch_size, 1), num_classes=num_classes)
val_preds = trained_model.predict(val_gen)
print(np.shape(val_preds))