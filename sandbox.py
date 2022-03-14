import data
import stack
import transformations as trans
import nibabel as nib
import torch as t
import os
import numpy as np
import nibabel as nib

filename = "iFIND00472_iFIND2_201901111210_PHILIPSJ2LF9B9_101_PIH1HR_Survey_32SENSE.s3.nii.gz"
t_image, t_affine, zooms = data.nii_to_torch(filename)

t_image_red = t_image[200:300,200:300,:]
data.show_slice(t_image_red)

#try registration
beta = 0.01
first_stack = stack.stack(t_image_red,t_affine, beta)
corners = first_stack.corners()
geometry = corners
n_voxels = t.tensor([50,5,50])
target = stack.volume(geometry,n_voxels)
    
    
for sl in range(0,4):
    #t_affine[2,3] += 1
    first_slice = stack.slice_2d(t_image_red[:,:,sl],t_affine)
    #for testing should give z-Axis
    first_slice.F = first_stack.F[:,:,sl]
    
    target_p_r = target.p_r
    p_s_tilde = target.p_s_tilde(first_slice.F)
    p_s = first_slice.p_s
    
    p_s_tilde_t = p_s_tilde.transpose(0,1).float()
    p_s_t = p_s.transpose(0,1).float()
    distance_matrix = t.cdist(p_s_t[:,:4], p_s_tilde_t)
    print(f'distance_matrix: {distance_matrix.shape}')
    
    closest_ps_tilde = t.min(distance_matrix,1).indices
    print(f'closest_ps_tilde: {closest_ps_tilde.shape}')
    
    distance_vector = t.sub(p_s[:3,:],p_s_tilde[:3,closest_ps_tilde])
    print(f'distance_vector {distance_vector.shape}')
    
    for i in range(0,closest_ps_tilde.shape[0]):
        target_index = closest_ps_tilde[i]
        target.p_r[4,target_index] += stack.PSN_Gauss(distance_vector[:,i])*p_s[4,i]
        #target.p_r[4,target_index] += p_s[4,i]
    
    target.update_X()

nft_img = data.torch_to_nii(target.X, target.affine)

filename = 'first_nifti'
data.save_nifti(nft_img,filename)