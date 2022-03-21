import utils
import stack
import volume
import torch as t
from scipy.spatial.transform import Rotation as R
import numpy as np
import time

def basic_reconstruction(resolution):
    filename = "s1_cropped.nii"
    folder = "data"
    t_image, t_affine, zooms = utils.nii_to_torch(folder, filename)
    
    #t_image_red = t_image
    t_image_red = t_image[100:200,100:210,:]
    utils.show_stack(t_image_red)
    
    beta = 0.01
    first_stack = stack.stack(t_image_red,t_affine, beta)
    corners = first_stack.corners()
    geometry = corners
    
    #intermediate solution
    n_voxels = t.ceil(t.multiply(t.tensor(t_image_red.shape),resolution))
    n_voxels[n_voxels < t.min(t.tensor(t_image_red.shape))] = t.min(t.tensor(t_image_red.shape)).item()
    n_voxels = n_voxels.int()
    #create target volume
    target = volume.volume()
    target.from_stack(geometry,n_voxels)
    target.reconstruct_stack(first_stack, time_it=True, batches = 5)
    #target.register_stack_euclid(first_stack)
    nft_img = utils.torch_to_nii(target.X, target.affine)
    folder = 'test_reconstruction'
    filename = 's1_smaller'
    utils.save_target(target,folder, filename)
    utils.save_nifti(nft_img,folder, filename)

def basic_2d_sampling(axis, rotation, I_x, I_y, I_z):
    filename = "s1_cropped.nii"
    folder = "data"
    t_image, t_affine, zooms = utils.nii_to_torch(folder, filename)
    #init sampling stack with rotation matrix
    I = t.zeros(I_x, I_y, I_z)
    r = R.from_euler(axis,rotation, degrees = True)
    t_affine = t.eye(4)
    t_affine[:3,:3] = t.tensor(r.as_matrix())
    sampling_stack = stack.stack(I,t_affine)
    #get volume to sample from
    folder = 'test_reconstruction'
    filename = 's1_cropped_complete'
    target_loaded = utils.open_target(folder, filename)
    
    sampling_stack.sample_from_volume(target_loaded)
    #sampling_2d(sampling_stack, target_loaded)
    utils.show_stack(sampling_stack.I)
    #target = volume.volume.from_stack(geometry,n_voxels)

if __name__ == '__main__':
    # resolution = 0.8
    # basic_reconstruction()
    
    axis = 'z'
    rotation = 45
    I_x, I_y, I_z = 150, 130, 6
    basic_2d_sampling(axis, rotation, I_x, I_y, I_z)
    
    
    
