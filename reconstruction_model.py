import torch as t
import volume
import stack
import utils
import monai


class Reconstruction(t.nn.Module):
    def __init__(self, target_volume:volume.volume, rec_stack:stack.stack, device):
        super().__init__()
        self.k = rec_stack.k
        self.rec_stack = rec_stack
        self.target_volume = target_volume
        self.rotations = t.nn.Parameter(t.zeros(3,self.k))
        self.translations = t.nn.Parameter(t.zeros(3,self.k))
        self.device = device
    def forward(self):
        #update Transition matrices of the stack
        for sl in range(0,self.k):
            T = utils.create_T(self.rotations[:,sl], self.translations[:,sl], self.device)
            self.rec_stack.T[:,:,sl] = T
        self.rec_stack.create_F()
        self.rec_stack.create_F_inv()
        self.target_volume.reconstruct_stack(self.rec_stack, batches = 50)
        
        return self.target_volume
    

class ReconstructionMonai(t.nn.Module):
    def __init__(self, k, device):
        super().__init__()
        self.k = k
        self.rotations = t.nn.Parameter(t.zeros(3,k))
        self.translations = t.nn.Parameter(t.zeros(3,k))
        self.device = device
        
    def forward(self, im_slices, target_dict, ground_spatial_dim):
        #transformed_slices = list()
        for sl in range(0,self.k):
            affine = utils.create_T(self.rotations[:,sl], self.translations[:,sl], self.device)
            affine_layer = monai.networks.layers.AffineTransform(mode = "bilinear", padding_mode = "zeros")
            im_slices[sl]["image"] = affine_layer(im_slices[sl]["image"], affine)
        target_dict = utils.reconstruct_3d_volume(im_slices, target_dict)
        #resample to initial resolution for loss
        target_dict["image"] = affine_layer(src = target_dict["image"],
                                            theta = t.eye(4),
                                            spatial_size = ground_spatial_dim)
        return target_dict
        

