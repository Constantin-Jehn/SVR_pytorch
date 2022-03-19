import data
import stack
import volume
import torch as t
import nibabel as nib
from scipy.spatial.transform import Rotation as R
import numpy as np
import time

def basic_reconstruction():
    filename = "s1_cropped.nii"
    folder = "data"
    t_image, t_affine, zooms = data.nii_to_torch(folder, filename)
    
    #t_image_red = t_image
    t_image_red = t_image[100:300,100:290,:]
    data.show_stack(t_image_red)
    
    beta = 0.01
    first_stack = stack.stack(t_image_red,t_affine, beta)
    corners = first_stack.corners()
    geometry = corners
    
    #intermediate solution
    resolution = 0.8
    n_voxels = t.ceil(t.multiply(t.tensor(t_image_red.shape),resolution))
    n_voxels[n_voxels < t.min(t.tensor(t_image_red.shape))] = t.min(t.tensor(t_image_red.shape)).item()
    n_voxels = n_voxels.int()
    #create target volume
    target = volume.volume()
    target.from_stack(geometry,n_voxels)
    target.reconstruct_stack(first_stack, time_it=True, batches = 5)
    #target.register_stack_euclid(first_stack)
    nft_img = data.torch_to_nii(target.X, target.affine)
    folder = 'test_reconstruction'
    filename = 's1_pstilde_new'
    data.save_target(target,folder, filename)
    data.save_nifti(nft_img,folder, filename)
    
def corners(p_r):
    """Function to get the "corners" of the stack in world coordinates to estimate the target volume size"""
    p_r_max = t.max(p_r[:3,:], dim = 1).values
    p_r_min = t.min(p_r[:3,:], dim = 1).values
    corners = t.stack((p_r_min,p_r_max)).transpose(0,1)
    return corners

def scale_ps_tilde(sampling_volume:volume.volume, sampling_stack:stack.stack):
    """Function to scale p_s_tilde properly during sampling R^(5,n_voxels_total,stack.k)"""
    i,j,k = t.ones(sampling_stack.k)*float("Inf"), t.ones(sampling_stack.k)*float("Inf"), t.ones(sampling_stack.k)*float("Inf")
    p_s_tilde = t.zeros(5,sampling_volume.p_r.shape[1],sampling_stack.k)
    for sl in range(0,sampling_stack.k):
        F = sampling_stack.F[:,:,sl]
        p_s_tilde[:,:,sl] = sampling_volume.p_s_tilde(F)
        i[sl], j[sl], k[sl] = t.min(p_s_tilde[0,:]), t.min(p_s_tilde[1,:]), t.min(p_s_tilde[2,:])
    i_min, j_min, k_min = t.min(i), t.min(j), t.min(k)
    #bring to start from 0
    p_s_tilde[0,:,:], p_s_tilde[1,:,:], p_s_tilde [2,:,:] =  p_s_tilde[0,:,:] - i_min, p_s_tilde[1,:,:] - j_min, p_s_tilde [2,:,:] - k_min
    #squish into sampling size
    i_max, j_max, k_max = t.max(p_s_tilde[0,:,:]), t.max(p_s_tilde[1,:,:]), t.max(p_s_tilde[2,:,:])
    i_slice, j_slice, k_slice = sampling_stack.I.shape[0], sampling_stack.I.shape[1], 0
    p_s_tilde[0,:,:], p_s_tilde[1,:,:], p_s_tilde[2,:,:] = p_s_tilde[0,:,:] * (i_slice/i_max), p_s_tilde[1,:,:] * (j_slice/j_max), p_s_tilde[2,:,:] *(k_slice/k_max)
    return p_s_tilde
    


def scale_ps_tilde_from_X(sampling_volume:volume.volume, sampling_stack:stack.stack):
    """Function to scale p_s_tilde properly during sampling R^(5,n_voxels_total,stack.k)"""
    p_s_tilde = sampling_volume.create_ps_from_X()
    #squish into sampling size
    i_max, j_max, k_max = t.max(p_s_tilde[0,:,:]), t.max(p_s_tilde[1,:,:]), t.max(p_s_tilde[2,:,:])
    i_slice, j_slice, k_slice = sampling_stack.I.shape[0], sampling_stack.I.shape[1], 0
    p_s_tilde[0,:,:], p_s_tilde[1,:,:], p_s_tilde[2,:,:] = p_s_tilde[0,:,:] * (i_slice/i_max), p_s_tilde[1,:,:] * (j_slice/j_max), p_s_tilde[2,:,:] 
    return p_s_tilde



def sampling_2d(sampling_stack,target_loaded):
    time_it = True
    batches = 5
    dist_threshold = 1.5
    p_s_tilde_scaled = scale_ps_tilde_from_X(target_loaded, sampling_stack)
    
    for sl in range(0,sampling_stack.k):
        p_s_tilde_t_complete = p_s_tilde_scaled[:,:,sl].transpose(0,1).float()
        p_s = sampling_stack.p_s[:,:,sl]
        p_s_t = p_s.transpose(0,1).float()
        #batchify not to run out of memory
        batch_size = round(p_s_tilde_t_complete.shape[0]/batches)
        lower_bound = 0
        upper_bound = batch_size
        
        for b in range(0,batches):
            #batch p_s_tilde
            if b < batches: 
                p_s_tilde_t = p_s_tilde_t_complete[lower_bound:upper_bound,:]
            else:
                p_s_tilde_t = p_s_tilde_t_complete[lower_bound:,:]
            #create distance tensor (n_voxels, n_pixels, 3) has difference in all three dimensions
            distance_tensor = t.abs(p_s_t[:,:3].unsqueeze(0) - p_s_tilde_t[:,:3].unsqueeze(1))
            #get indices that match from voxel to pixel [(voxel_index),(pixel_index)]
            indices = t.where(t.all(distance_tensor < dist_threshold,2))
            #indices = t.where(t.norm(distance_tensor,dim=2) < 1.5)
            #number of matches
            length = list(indices[0].shape)[0]
            #extract relevant p_s and calculate PSN vectorized
            #relevant_p_r = target_loaded.p_r[4,indices[0]]
            relevant_p_r = p_s_tilde_scaled[4, indices[0], sl]
            #relevant_p_s = p_s[4,indices[1]]
            relevant_dist = distance_tensor[indices[0],indices[1],:]
            gauss = volume.PSN_Gauss_vec(relevant_dist)
            value = t.multiply(gauss, relevant_p_r)
            
            #add values to corresponding pixesl
            for voxel in range(0,length):
                sampling_stack.p_s[4,indices[1][voxel],sl] += value[voxel]
            
            lower_bound += batch_size
            upper_bound += batch_size
        print(f'slice {sl} sampled \n ')

if __name__ == '__main__':
    #basic_reconstruction()

    filename = "s1_cropped.nii"
    folder = "data"
    t_image, t_affine, zooms = data.nii_to_torch(folder, filename)
    
    I = t.zeros(150,150,5)
    # #take affine from
    t_affine = t.eye(4)
    sampling_stack = stack.stack(I,t_affine)
    #get volume to sample from
    folder = 'test_reconstruction'
    filename = 's1_cropped_complete'
    target_loaded = data.open_target(folder, filename)
    
    sampling_2d(sampling_stack, target_loaded)
    data.show_stack(sampling_stack.create_I_from_ps())
    # #target = volume.volume.from_stack(geometry,n_voxels)
    

    
    
