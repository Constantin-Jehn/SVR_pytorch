import torch as t
import monai
from monai.transforms import (
    AddChanneld,
    LoadImage,
    LoadImaged,
    Orientationd,
    Rand3DElasticd,
    RandAffined,
    Spacingd,
    ToTensord,
    RandAffine
)
from copy import deepcopy


class Reconstruction(t.nn.Module):
    def __init__(self, n_slices:int, device):
        super().__init__()
        
        self.device = device
        self.n_slices = n_slices
        self.rotations = t.nn.ParameterList([t.nn.Parameter(t.zeros(3, device = self.device)) for i in range(n_slices)])
        self.translations = t.nn.ParameterList([t.nn.Parameter(t.zeros(3, device = self.device)) for i in range(n_slices)])
        self.affine_layer = monai.networks.layers.AffineTransform(mode = "bilinear",  normalized = True, padding_mode = "zeros")
    
    def forward(self, im_slices, ground_meta, fixed_image_meta, transform_to_fixed = True):
        
        
        resampler = monai.transforms.ResampleToMatch()
        
        affines = self.create_T(self.rotations[0], self.translations[0]).unsqueeze(0)
        for sli in range(1,self.n_slices):
            affines = t.cat((affines,self.create_T(self.rotations[sli], self.translations[sli]).unsqueeze(0)),0)
            
        im_slices = self.affine_layer(im_slices, affines)
        
        
        if transform_to_fixed:
            transformed_size = (self.n_slices,1) + tuple(fixed_image_meta["spatial_shape"])
            transformed_slices = t.zeros(transformed_size)
            
            for sli in range(0,self.n_slices):
                transformed_slices[sli,:,:,:,:], _ = resampler(im_slices[sli,:,:,:,:],src_meta = ground_meta, 
                                              dst_meta = fixed_image_meta, padding_mode = "zeros")
        else:
            transformed_slices = im_slices
            
        return transformed_slices  
    
    def rotation_matrix(self, angles):
        """
        Returns a rotation matrix for given angles.
        Own implementation to assure the possibility of a computational graph
        for update of parameters

        Parameters
        ----------
        angles : list
            desired angles in radian

        Returns
        -------
        torch.tensor
            rotation matrix

        """
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
        

    def create_T(self,rotations, translations):
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
        rotation = self.rotation_matrix(rotations).to(self.device)
        bottom = t.tensor([0,0,0,1],device = self.device)
        trans = t.cat((rotation,translations.unsqueeze(1)),dim=1).to(self.device)
        T = t.cat((trans,bottom.unsqueeze(0)),dim = 0)
        return T
            
        
        