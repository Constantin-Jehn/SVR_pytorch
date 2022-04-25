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
import pytorch3d as p3d


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
        else:
            transformed_slices = im_slices
            
        return transformed_slices  

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
            
        
        