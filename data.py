import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import torch as t

import transformations as trans

from stack import stack

def nii_to_torch(filename):
    """opens nifti file from /data and return data and affine as torch tensors"""
    path = os.path.join("data", filename)
    img = nib.load(path)
    epi_image = t.tensor(img.get_fdata())
    affine = t.tensor(img.affine)
    return epi_image, affine

def torch_to_nii(data,affine):
    data = data.numpy().astype('int16')
    affine = affine.numpy().astype('int16')
    new_img = nib.Nifti2Image(data,affine)
    return new_img

def show_slice(t_image):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, t_image.shape[2])
    for i in range(t_image.shape[2]):
      axes[i].imshow(t.transpose(t_image[:,:,i],1,0), cmap="gray", origin="lower")
    return 0
      


