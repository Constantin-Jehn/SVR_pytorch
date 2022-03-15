import data
import structures
import transformations as trans
import nibabel as nib
import torch as t
import os
import numpy as np
import nibabel as nib

filename = "iFIND00472_iFIND2_201901111210_PHILIPSJ2LF9B9_101_PIH1HR_Survey_32SENSE.s3.nii.gz"
t_image, t_affine, zooms = data.nii_to_torch(filename)

t_image_red = t_image
t_image_red = t_image[200:400,200:400,:]
data.show_stack(t_image_red)

#try registration
beta = 0.01
first_stack = structures.stack(t_image_red,t_affine, beta)
corners = first_stack.corners()
geometry = corners

resolution = 0.5
[l_x,l_y,l_z] = geometry[:,0] - geometry[:,1]
n_voxels = t.ceil(t.abs(t.div(t.tensor([l_x,l_y,l_z]), resolution))).int()

#n_voxels = t.tensor([80,5,80])
target = structures.volume(geometry,n_voxels)

target.register_stack(first_stack)

nft_img = data.torch_to_nii(target.X, target.affine)

filename = 'first_nifti_resolution'
data.save_nifti(nft_img,filename)
