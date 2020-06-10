import numpy as np
from keras import backend as K

def dsc_loss_fcn(y_true, y_pred):
    smooth = 1
    y_true_flat, y_pred_flat = K.flatten(y_true), K.flatten(y_pred)
    dice_nom = 2 * K.sum(y_true_flat * y_pred_flat)
    dice_denom = K.sum(K.square(y_true_flat) + K.square(y_pred_flat)) 
    dice_coef = (dice_nom + smooth) / (dice_denom + smooth)
    return 1 - dice_coef

# from https://github.com/keras-team/keras/issues/9395
def dice_coef_ignore_bg(y_true, y_pred):
    '''shape of y_true and y_pred are [batch, x, y, z, n_classes]'''
    smooth = 1

    y_true = y_true[:,:,:,:,1:]
    y_pred = y_pred[:,:,:,:,1:]

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_ignore_bg_loss(y_true, y_pred):
    return 1-dice_coef_ignore_bg(y_true,y_pred)

#calculates dice considering an input with a single class
# from https://stackoverflow.com/questions/53926178/is-there-a-way-to-output-a-metric-with-several-values-in-keras
def dice_single(true,pred):
    '''the shape of true and pred are [batch, x, y, z]'''
    smooth = 1
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_for_class(index):
    def dice_inner(true, pred):

        #get only the desired class
        true = true[:,:,:,:,index]
        pred = pred[:,:,:,:,index]

        #return dice per class
        return dice_single(true, pred)
    return dice_inner

# Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
# -> the score is computed for each class separately and then summed
# alpha=beta=0.5 : dice coefficient
# alpha=beta=1   : tanimoto coefficient (also known as jaccard)
# alpha+beta=1   : produces set of F*-scores
# implemented by E. Moebel, 06/04/18
# https://github.com/keras-team/keras/issues/9395
def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta  = 0.5
    
    ones = K.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones-y_true
    
    num = K.sum(p0*g0, (0,1,2,3))
    den = num + alpha*K.sum(p0*g1,(0,1,2,3)) + beta*K.sum(p1*g0,(0,1,2,3))
    
    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
    
    Ncl = K.cast(K.shape(y_true)[-1], 'float32') #the number of classes should be the last channel
    return Ncl-T