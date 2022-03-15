import data
import structures
import transformations as trans
import nibabel as nib
import torch as t
import os
import numpy as np
import nibabel as nib
import time

filename = "iFIND00472_iFIND2_201901111210_PHILIPSJ2LF9B9_101_PIH1HR_Survey_32SENSE.s1.nii.gz"
t_image, t_affine, zooms = data.nii_to_torch(filename)


t_image_red = t_image[200:350,200:350,:]
data.show_stack(t_image_red)

#try registration
beta = 0.01
first_stack = structures.stack(t_image_red,t_affine, beta)
corners = first_stack.corners()
geometry = corners

# resolution = 0.5
# [l_x,l_y,l_z] = geometry[:,0] - geometry[:,1]
# n_voxels = t.ceil(t.abs(t.multiply(t.tensor([l_x,l_y,l_z]), resolution)))

# n_voxels[n_voxels < t.min(t.tensor(t_image_red.shape))] = t.min(t.tensor(t_image_red.shape)).item()
# n_voxels = n_voxels.int()

n_voxels = t.tensor([80,80,7])
target = structures.volume(geometry,n_voxels)

target.register_stack(first_stack, time_it=True)
nft_img = data.torch_to_nii(target.X, t.eye(4))

folder = 'test_reconstruction'
filename = 's1_correct_vec'
data.save_nifti(nft_img,folder, filename)



