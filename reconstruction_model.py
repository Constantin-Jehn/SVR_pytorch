import torch as t
import utils
import monai
from monai.transforms import (
    AddChanneld)

class ReconstructionMonai(t.nn.Module):
    def __init__(self, k, device):
        super().__init__()
        self.k = k
        self.rotations = t.nn.Parameter(t.zeros(3,k))
        self.translations = t.nn.Parameter(t.zeros(3,k))
        self.affine_layer = monai.networks.layers.AffineTransform(mode = "bilinear", padding_mode = "zeros")
        self.device = device
        
    def forward(self, im_slices, target_dict, ground_spatial_dim):
        #transformed_slices = list()
        for sl in range(0,self.k):
            affine = utils.create_T(self.rotations[:,sl], self.translations[:,sl], self.device)
            im_slices[sl]["image"] = self.affine_layer(im_slices[sl]["image"], affine)
        
        target_dict = utils.reconstruct_3d_volume(im_slices, target_dict)
        # print("target in low res")
        # plt.imshow(t.squeeze(target_dict["image"])[12,:,:].detach().numpy(), cmap="gray")
        # plt.show()
        return target_dict
        