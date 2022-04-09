import torch as t
import utils
import monai
from monai.transforms import (
    AddChanneld)

class ReconstructionMonai(t.nn.Module):
    def __init__(self, k, device, mode):
        super().__init__()
        self.k = k
        self.rotations = t.nn.Parameter(t.zeros(3,k))
        self.translations = t.nn.Parameter(t.zeros(3,k))
        self.affine_layer = monai.networks.layers.AffineTransform(mode = mode,  normalized = True, padding_mode = "zeros")
        self.device = device
        
    def forward(self, im_slices, target_dict):
        #transformed_slices = list()
        for sl in range(0,self.k):
            affine = utils.create_T(self.rotations[:,sl], self.translations[:,sl], self.device)
            im_slices[sl]["image"] = self.affine_layer(im_slices[sl]["image"], affine)
        target_dict = utils.reconstruct_3d_volume(im_slices, target_dict)
        # print("target in low res")
        # plt.imshow(t.squeeze(target_dict["image"])[12,:,:].detach().numpy(), cmap="gray")
        # plt.show()
        return target_dict

class Reconstruction(t.nn.Module):
    def __init__(self,n_stacks:int,n_slices_max:int, device):
        super().__init__()
        self.device = device
        self.n_stacks = n_stacks
        self.rotations = t.nn.Parameter(t.zeros(3,n_stacks,n_slices_max))
        self.translations = t.nn.Parameter(t.zeros(3,n_stacks,n_slices_max))
        self.affine_layer = monai.networks.layers.AffineTransform(mode = "bilinear",  normalized = True, padding_mode = "zeros")
    
    def forward(self, im_slices):
        if len(im_slices) != self.n_stacks:
            assert("slice dimensions does not match pre-defined number of stacks.")
        for sta in range(0,self.n_stacks):
            slices = im_slices[sta]
            n_slices = len(slices)
            for sli in range(0,n_slices):
                affine = utils.create_T(self.rotations[:,sta,sli],self.translations[:,sta,sli], self.device)
                slices[sli] = self.affine_layer(slices[sli], affine)
            im_slices[sta] = slices
            
        return im_slices   
            
            
        

    #batched:
    # def forward(self, im_slices, target_dict, ground_spatial_dim):
    #     #transformed_slices = list()
    #     batched_affine = utils.create_T(self.rotations[:,0], self.translations[:,0], self.device)
    #     batched_images = im_slices[0]["image"]
        
    #     for sl in range(1,self.k):
    #         affine = utils.create_T(self.rotations[:,sl], self.translations[:,sl], self.device)
    #         t.stack((batched_affine, affine), dim = 1)
    #         image = im_slices[sl]["image"]
    #         t.stack((batched_images,image), dim = 0)
        
        
        
    #     batched_images_transformed = self.affine_layer(im_slices[sl]["image"], affine)
        
    #     for sl in range(0,self.k):
    #         im_slices[sl]["image"] = batched_images_transformed[:,:,sl,:,:]
        
    #     target_dict = utils.reconstruct_3d_volume(im_slices, target_dict)
    #     # print("target in low res")
    #     # plt.imshow(t.squeeze(target_dict["image"])[12,:,:].detach().numpy(), cmap="gray")
    #     # plt.show()
    #     return target_dict
        