import numpy as np 
import nibabel as nib 
import keras
import os
import numbers
from sklearn.utils import check_array, check_random_state
from numpy.lib.stride_tricks import as_strided
from itertools import product
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img

def load_nii(path): #use for both input and label loading
    
    nii_struct = nib.load(path)
    nii_img = nii_struct.get_fdata()
    
    return( nii_img )

def write_nii(img, path):
    nii_to_write = nib.Nifti1Image(img, np.eye(4))
    nib.save(nii_to_write, path)
    return path


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



def filter_patches(patch_masks):
    '''Removes patches that are entirely background (class 0)
    Assumes input is patch masks of shape [batch, x, y, z, n_classes]'''
    s = np.sum(patch_masks[:,:,:,:,1:],axis=(1,2,3,4))
    return s > 0

class niftiDataGen(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, patch_size, extraction_step, input_img_paths, target_img_paths, n_classes):
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.extraction_step = extraction_step
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.n_classes = n_classes

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        
        #trim a random bit off the image to vary patch centers epoch to epoch
        r = np.random.randint(0, self.patch_size[0]/2, size=(3,))

        for j, path in enumerate(batch_input_img_paths):
            whole_img = load_nii(path)
            whole_img = whole_img[r[0]:,r[1]:,r[2]:]

            X = extract_patches(whole_img, patch_shape=self.patch_size, extraction_step=self.extraction_step)
            X = X.reshape([-1] + list(self.patch_size))
            X = np.expand_dims(X, axis=4) #now shape is [batch, x, y, z, n_channels]

        for j, path in enumerate(batch_target_img_paths):
            whole_mask = load_nii(path)
            whole_mask = whole_mask[r[0]:,r[1]:,r[2]:]

            y = extract_patches(whole_mask, patch_shape=self.patch_size, extraction_step=self.extraction_step)
            y = y.reshape([-1] + list(self.patch_size))
            y = keras.utils.to_categorical(y) #now shape should be [batch, x, y, z, n_classes]

        #remove patches that are all blank
        ind = filter_patches(y)
        Xf = X[ind,:,:,:]
        yf = y[ind,:,:,:]

        return Xf, yf