import data
import stack
import transformations as trans
import nibabel as nib
import torch as t
import os
import numpy as np

filename = "iFIND00472_iFIND2_201901111210_PHILIPSJ2LF9B9_101_PIH1HR_Survey_32SENSE.s3.nii.gz"
t_image, t_affine, zooms = data.nii_to_torch(filename)

beta = 0.01
first_stack = stack.stack(t_image,t_affine, beta)

W_s = np.array([[2,1],[1,1]])
T = np.array([[3,1],[1,3]])
W_r = np.array([[1,2],[1,4]])

W_s = t.tensor([[2,1],[1,1]])
T = t.tensor([[3,1],[1,3]])
T_2 = t.tensor([[1,0],[0,1]])
W_r = t.tensor([[1.0,2.0],[1.0,4.0]])

first_stack.T = t.stack((T,T_2), dim=2)
first_stack.W_s = W_s.unsqueeze(2).repeat(1,1,2)
first_stack.affine = W_r


F = first_stack.F()
F_inv = first_stack.F_inv()




# new_nifti = data.torch_to_nii(t_image,t_affine)

# data = t_image.numpy().astype('int16')
# affine = t_affine.numpy().astype('int16')

# #data = np.ones((32, 32, 15, 100), dtype=np.int16)
# #affine = np.eye(4,dtype=np.int16)

# img = nib.Nifti1Image(data, affine)

# img.header.get_xyzt_units()
# nib.save(img, os.path.join('test.nii.gz')) 