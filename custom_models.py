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
from monai.transforms.utils import (
    create_rotate,
    create_translate
    )
from copy import deepcopy

import numpy as np


class Reconstruction(t.nn.Module):
    def __init__(self, n_slices:int, device):
        super().__init__()
        
        self.device = device
        self.n_slices = n_slices
        self.rotations = t.nn.ParameterList([t.nn.Parameter(t.zeros(3, device = self.device)) for i in range(n_slices)])
        self.translations = t.nn.ParameterList([t.nn.Parameter(t.zeros(3, device = self.device)) for i in range(n_slices)])
        self.affine_layer = monai.networks.layers.AffineTransform(mode = "bilinear",  normalized = True, padding_mode = "zeros")
    
    def forward(self, im_slices, ground_meta, fixed_image_meta, transform_to_fixed = True, mode = "bilinear"):
        
        resampler = monai.transforms.ResampleToMatch(mode = mode)
        
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
            return transformed_slices  
        
        else:
            return im_slices
    
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
        #rotation = self.rotation_matrix(rotations).to(self.device)
        rotation_tensor = monai.transforms.utils.create_rotate(3, rotations, device = self.device,  backend="torch")
        translation_tensor = monai.transforms.utils.create_translate(3, translations, device = self.device, backend="torch")
        T = t.matmul(rotation_tensor,translation_tensor)
        return T
            

        
class Volume_to_Slice(t.nn.Module):
    def __init__(self, n_slices:int, device):
        super().__init__()
        
        self.device = device
        self.n_slices = n_slices
        self.rotations = t.nn.ParameterList([t.nn.Parameter(t.zeros(3, device = self.device)) for i in range(n_slices)])
        self.translations = t.nn.ParameterList([t.nn.Parameter(t.zeros(3, device = self.device)) for i in range(n_slices)])
        self.affine_layer = monai.networks.layers.AffineTransform(mode = "bilinear",  normalized = True, padding_mode = "zeros")
    
    def forward(self, fixed_image_image, fixed_image_meta, local_stack_meta, mode = "bilinear"):
        """
        fixed_image_image: image_tensor of registration target/volume
        fixed_image_meta: dictionary with meta data of fixed_image
        local_stack_meta: dictionary with meta data of local stack
        mode: interpolation mode for resampling

        returns:
        fixed_image_tran: tensor containing the fixed images transformed by the inverse affines of each slice, hence the corresponding layer 
        is a simulated slice
        affines: affine transformation matrices for each slice
        """
        resampler = monai.transforms.ResampleToMatch(mode = mode)
        add_channel = AddChanneld(keys=["image"])
        
        
        #resample fixed image to local stack and repeat n_slices time for batch-format
        fixed_image_image = fixed_image_image.squeeze().unsqueeze(0)
        fixed_image_image, fixed_image_meta = resampler(fixed_image_image,src_meta=fixed_image_meta,
                         dst_meta=local_stack_meta)
        fixed_image_meta["spatial_shape"] = np.array(list(fixed_image_image.shape)[1:])
        
        fixed_image_image = fixed_image_image.unsqueeze(0)
        
        fixed_image_image_batch = fixed_image_image.repeat(self.n_slices,1,1,1,1)
        
        #create affines and inv affines
        aff = self.create_T(self.rotations[0], self.translations[0])
        inv_aff = t.linalg.inv(aff)
        affines = aff.unsqueeze(0)
        inv_affines = inv_aff.unsqueeze(0)
        
        for sli in range(1,self.n_slices):
            aff = self.create_T(self.rotations[sli], self.translations[sli])
            inv_aff = t.linalg.inv(aff)
    
            affines = t.cat((affines,aff.unsqueeze(0)),0)
            inv_affines = t.cat((inv_affines,inv_aff.unsqueeze(0)),0)

        #slice simulation by transforming fixed image by inv affine --> layer are only 2 dimensional and align with the slices
        # note that fixed image was resamples to local stack so dimensionality matches and we can forward the transformed fixed images directly to the loss
        fixed_image_tran = self.affine_layer(fixed_image_image_batch, inv_affines)

        return fixed_image_tran, affines

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
        #rotation = self.rotation_matrix(rotations).to(self.device)
        rotation_tensor = monai.transforms.utils.create_rotate(3, rotations, device = self.device,  backend="torch")
        translation_tensor = monai.transforms.utils.create_translate(3, translations, device = self.device, backend="torch")
        T = t.matmul(rotation_tensor,translation_tensor)
        return T
            
        
        