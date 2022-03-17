import data
import stack
import volume
import torch as t

def basic_reconstruction():
    filename = "s1_cropped.nii"
    t_image, t_affine, zooms = data.nii_to_torch(filename)
    
    #t_image_red = t_image
    t_image_red = t_image[150:350,150:350,:]
    data.show_stack(t_image_red)
    
    beta = 0.01
    first_stack = stack.stack(t_image_red,t_affine, beta)
    corners = first_stack.corners()
    geometry = corners
    
    #intermediate solution
    resolution = 0.5
    n_voxels = t.ceil(t.multiply(t.tensor(t_image_red.shape),resolution))
    n_voxels[n_voxels < t.min(t.tensor(t_image_red.shape))] = t.min(t.tensor(t_image_red.shape)).item()
    n_voxels = n_voxels.int()
    #create target volume
    target = volume.volume(geometry,n_voxels)
    
    target.reconstruct_stack(first_stack, time_it=True, batches = 3)
    #target.register_stack_euclid(first_stack)
    
    nft_img = data.torch_to_nii(target.X, t.eye(4))
    folder = 'test_reconstruction'
    filename = 's1_batched'
    data.save_nifti(nft_img,folder, filename)
    


if __name__ == '__main__':
    basic_reconstruction()
    
