import data
import stack
import transformations as trans
import nibabel as nib
import torch as t
import os

filename = "iFIND00472_iFIND2_201901111210_PHILIPSJ2LF9B9_101_PIH1HR_Survey_32SENSE.s1.nii.gz"
t_image, t_affine = data.nii_to_torch(filename)

beta = 0.01
first_stack = stack.stack(t_image,t_affine, beta)
rank_D= first_stack.rank_D()
er = first_stack.within_stack_error()
F = first_stack.F()
F_inv = first_stack.F_inv()

##To do: check F and F_inv

data.show_slice(t_image)
M = t_affine[:3,:3]
abc = t_affine[:3,3]
epi_center = (t.tensor(t_image.shape)-1)/2

RAS_center = trans.f(t_image,t_affine)

t_image = trans.R_z(t_image,0,180)
data.show_slice(t_image)

t_image = trans.T_x(t_image,1,150)
data.show_slice(t_image)

t_image = trans.T_y(t_image,2,-150)
data.show_slice(t_image)




new_nifti = data.torch_to_nii(t_image,t_affine)

data = t_image.numpy().astype('int16')
affine = t_affine.numpy().astype('int16')

#data = np.ones((32, 32, 15, 100), dtype=np.int16)
#affine = np.eye(4,dtype=np.int16)

img = nib.Nifti1Image(data, affine)

img.header.get_xyzt_units()
nib.save(img, os.path.join('test.nii.gz')) 