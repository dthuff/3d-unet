import numpy as np 
import nibabel as nib 
import keras
import os
import numbers
from sklearn.utils import check_array, check_random_state
from numpy.lib.stride_tricks import as_strided
from itertools import product

def load_nii(path): #use for both input and label loading
    
    nii_struct = nib.load(path)
    nii_img = nii_struct.get_fdata()
    
    return( nii_img )


def preprocess_input( image ):
    
    '''--- Rescale Image
    --- Rotate Image
    --- Resize Image
    --- Flip Image
    --- PCA etc.'''
    
    return( image )

def extract_patches(arr, patch_shape=8, extraction_step=1): #from https://github.com/konopczynski/Vessel3DDL/blob/master/scripts/utils/patches_3d.py
    """Extracts patches of any n-dimensional array in place using strides.
    Given an n-dimensional array it will return a 2n-dimensional array with
    the first n dimensions indexing patch position and the last n indexing
    the patch content. This operation is immediate (O(1)). A reshape
    performed on the first n dimensions will cause numpy to copy data, leading
    to a list of extracted patches.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    arr : ndarray
        n-dimensional array of which patches are to be extracted
    patch_shape : integer or tuple of length arr.ndim
        Indicates the shape of the patches to be extracted. If an
        integer is given, the shape will be a hypercube of
        sidelength given by its value.
    extraction_step : integer or tuple of length arr.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.
    Returns
    -------
    patches : strided ndarray
        2n-dimensional array indexing patches on first n dimensions and
        containing patches on the last n dimensions. These dimensions
        are fake, but this way no data is copied. A simple reshape invokes
        a copying operation to obtain a list of patches:
        result.reshape([-1] + list(patch_shape))
    """

    arr_ndim = arr.ndim

    if isinstance(patch_shape, numbers.Number):
        patch_shape = tuple([patch_shape] * arr_ndim)
    if isinstance(extraction_step, numbers.Number):
        extraction_step = tuple([extraction_step] * arr_ndim)

    patch_strides = arr.strides

    slices = [slice(None, None, st) for st in extraction_step]
    #indexing_strides = arr[slices].strides
    indexing_strides = arr[tuple(slices)].strides

    patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                           np.array(extraction_step)) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))

    patches = as_strided(arr, shape=shape, strides=strides)
    return patches

def nifti_train_generator(img_dir, mask_dir, batch_size, input_size, extraction_step): #based on amlarraz comment https://github.com/keras-team/keras/issues/3059
    list_images = os.listdir(img_dir)
    list_mask = os.listdir(mask_dir)
    #shuffle(list_images) #Randomize the choice of batches
    ids_train_split = range(len(list_images))
    while True:
         for start in range(0, len(ids_train_split), batch_size):
            X = np.empty((batch_size, *input_size, 1))
            y = np.empty((batch_size, *input_size, 1))

            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[start:end]
            for id in ids_train_batch: #needs to draw patches
                whole_img = load_nii(os.path.join(img_dir, list_images[id]))
                whole_mask = load_nii(os.path.join(mask_dir, list_mask[id]))

                X = extract_patches(whole_img, patch_shape=input_size, extraction_step=extraction_step)
                y = extract_patches(whole_mask, patch_shape=input_size, extraction_step=extraction_step)

                X = X.reshape([-1] + list(input_size))
                y = y.reshape([-1] + list(input_size))

                X = np.expand_dims(X, axis=4) #shape is [batch, x, y, z]
                y = keras.utils.to_categorical(y) #now shape should be [batch, x, y, z, n_class]
                

            yield X, y


# Not using this right now
class niftiDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(64,64,64), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # X[i,] = np.load('data/' + ID + '.npy')
            X[i,] = load_nii('data/image/' + ID + '.nii.gz')

            # Store class
            y[i] = load_nii('data/mask/' + ID + '.nii.gz')

        return X, y