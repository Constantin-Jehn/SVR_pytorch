import utils
import stack
import volume
import torch as t
from scipy.spatial.transform import Rotation as R
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader
import reconstruction_model

def basic_reconstruction(resolution):
    filename = "10_3T_nody_002.nii.gz"
    folder = "sample_data"
    t_image, t_affine, zooms = utils.nii_to_torch(folder, filename)
    
    t_image_red = t_image
    t_image_red = t_image[:,:,:10]
    utils.show_stack(t_image_red)
    
    beta = 0.01
    first_stack = stack.stack(t_image_red,t_affine, beta)
    corners = first_stack.corners()
    geometry = corners
    
    #intermediate solution
    #resolution = 0.5
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
    filename = 'sample_data'
    utils.save_target(target,folder, filename)
    utils.save_nifti(nft_img,folder, filename)

def basic_2d_sampling(rotations, I_x, I_y, I_z):
    filename = "s1_cropped.nii"
    folder = "data"
    t_image, t_affine, zooms = utils.nii_to_torch(folder, filename)
    #init sampling stack with rotation matrix
    I = t.zeros(I_x, I_y, I_z)
    t_affine = t.eye(4)
    t_affine = utils.create_T(rotations, t.zeros(3))
    sampling_stack = stack.stack(I,t_affine)
    #get volume to sample from
    folder = 'test_reconstruction'
    filename = 's1_cropped_complete'
    target_loaded = utils.open_target(folder, filename)
    
    sampling_stack.sample_from_volume(target_loaded)
    #sampling_2d(sampling_stack, target_loaded)
    utils.show_stack(sampling_stack.I)
    #target = volume.volume.from_stack(geometry,n_voxels)
    
def check_ncc():
    folder = 'test_reconstruction'
    filename = 's1_cropped_complete'
    volume1 = utils.open_target(folder, filename)
    volume2 = utils.open_target(folder, filename)
    ncc = utils.ncc(volume1, volume2)
    print(f'ncc identical: {ncc}')
    disturbed = volume2.p_r[4,:] * (1+t.rand(volume2.p_r.shape[1]) *100000)  
    volume2.p_r[4,:] = disturbed
    print(f'ncc disturbed: {utils.ncc(volume1, volume2)}')
    
def optimize(resolution):
    #create stack
    filename = "s1_cropped.nii"
    folder = "data"
    t_image, t_affine, zooms = utils.nii_to_torch(folder, filename)
    t_image_red = t_image[100:150,100:160,:]
    first_stack = stack.stack(t_image_red,t_affine, add_init_offset=False)
    geometry = first_stack.corners()
    #create target volume
    n_voxels = t.ceil(t.multiply(t.tensor(t_image_red.shape),resolution))
    n_voxels[n_voxels < t.min(t.tensor(t_image_red.shape))] = t.min(t.tensor(t_image_red.shape)).item()
    n_voxels = n_voxels.int()
    #create target volume
    target = volume.volume()
    target.from_stack(geometry,n_voxels)
    
    model = reconstruction_model.Reconstruction(target,first_stack)
    optimizer = t.optim.SGD(model.parameters(), lr = 0.01)
    
    #t.autograd.set_detect_anomaly(True)
    for i in range(0,5):
        target_volume = model()
        loss = utils.ncc_within_volume(target_volume)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        print(f'epoch: {i} ncc: {loss}')
        optimizer.step()

if __name__ == '__main__':
    resolution = 0.3
    basic_reconstruction(resolution)
    # rotation = t.tensor([0,0,45])
    # I_x, I_y, I_z = 150, 130, 6
    # basic_2d_sampling(rotation, I_x, I_y, I_z)
    #check ncc
    # folder = 'test_reconstruction'
    # filename = 's1_cropped_complete'
    # volume1 = utils.open_target(folder, filename)
    # ncc_within = utils.ncc_within_volume(volume1)
    # print(ncc_within)
    
    # print(utils.create_T([0,0,90],[1,2,0]))
    #resolution = 0.3
    #optimize(resolution)

    
    

    
    
