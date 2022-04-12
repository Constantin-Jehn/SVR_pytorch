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
            affine = self.create_T(self.rotations[:,sl], self.translations[:,sl], self.device)
            im_slices[sl]["image"] = self.affine_layer(im_slices[sl]["image"], affine)
        target_dict = utils.reconstruct_3d_volume(im_slices, target_dict)
        # print("target in low res")
        # plt.imshow(t.squeeze(target_dict["image"])[12,:,:].detach().numpy(), cmap="gray")
        # plt.show()
        return target_dict

class Reconstruction(t.nn.Module):
    def __init__(self, n_slices:int, device):
        super().__init__()
        self.device = device
        self.n_slices = n_slices
        self.rotations = t.nn.Parameter(t.zeros(3,n_slices))
        self.translations = t.nn.Parameter(t.zeros(3,n_slices))
        self.affine_layer = monai.networks.layers.AffineTransform(mode = "bilinear",  normalized = True, padding_mode = "zeros")
    
    def forward(self, im_slices):
        for sli in range(0,self.n_slices):
            affine = self.create_T(self.rotations[:,sli],self.translations[:,sli], self.device)
            im_slices[sli] = self.affine_layer(im_slices[sli], affine)
        return im_slices  
    
    def rotation_matrix(self, angles):
        s = t.sin(angles)
        c = t.cos(angles)
        rot_x = t.cat((t.tensor([1,0,0]),
                      t.tensor([0,c[0],-s[0]]),
                      t.tensor([0,s[0],c[0]])), dim = 0).reshape(3,3)
        
        rot_y = t.cat((t.tensor([c[1],0,s[1]]),
                      t.tensor([0,1,0]),
                      t.tensor([-s[1],0,c[1]])),dim = 0).reshape(3,3)
        
        rot_z = t.cat((t.tensor([c[2],-s[2],0]),
                      t.tensor([s[2],c[2],0]),
                      t.tensor([0,0,1])), dim = 0).reshape(3,3)
        return t.matmul(t.matmul(rot_z, rot_y),rot_x)
        

    def create_T(self,rotations, translations, device):
        """
        Parameters
        ----------
        rotations : t.tensor (1x3)
            convention XYZ
        translations : t.tensor (1x3)
            translations

        Returns
        -------
        T : TYPE
            DESCRIPTION.

        """
        rotation = self.rotation_matrix(rotations).to(device)
        bottom = t.tensor([0,0,0,1]).to(device)
        trans = t.cat((rotation,translations.unsqueeze(1)),dim=1).to(device)
        T = t.cat((trans,bottom.unsqueeze(0)),dim = 0)
        return T
            
            
        

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
        