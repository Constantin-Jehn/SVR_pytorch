import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from nibabel.testing import data_path
from nibabel.affines import apply_affine
import torch as t

def nii_to_torch(filename):
    path = os.path.join("data", filename)
    img = nib.load(path)
    epi_image = t.tensor(img.get_fdata())
    affine = t.tensor(img.affine)
    return epi_image, affine

def show_slice(t_image):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, t_image.shape[2])
    for i in range(t_image.shape[2]):
      axes[i].imshow(t.transpose(t_image[:,:,i],1,0), cmap="gray", origin="lower")
      
      
def f(t_image, t_affine):
    """Function to map from Voxel coordinates to reference coordinates """
    M = t_affine[:3,:3]
    abc = t_affine[:3,3]
    epi_center = ((t.tensor(t_image.shape)-1)/2).double()
    print(M.shape)
    print(epi_center)
    print(abc)
    return t.linalg.multi_dot([M,epi_center]) + abc     

def f_inv(t_image, t_affine):
    """Function to map from reference coordinates to Voxel """
    inv = t.linalg.inv(t_affine)
    epi_center = t.cat((((t.tensor(t_image.shape)-1)/2).double(),t.tensor([1])))
    voxel = t.linalg.multi_dot(inv,epi_center)
    return voxel.chunk(3)[0]
    


filename = "iFIND00472_iFIND2_201901111210_PHILIPSJ2LF9B9_101_PIH1HR_Survey_32SENSE.s1.nii.gz"
t_image, t_affine = nii_to_torch(filename)

show_slice(t_image)

M = t_affine[:3,:3]
abc = t_affine[:3,3]
epi_center = (t.tensor(t_image.shape)-1)/2

RAS_center = f(t_image,t_affine)

voxel_center = f_inv()
