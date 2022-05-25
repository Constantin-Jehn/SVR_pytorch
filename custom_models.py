import torch as t
import monai
from monai.transforms import (
    AddChanneld
)
from monai.transforms.utils import (
    create_rotate,
    create_translate
    )
from copy import deepcopy

import numpy as np

import torchio as tio

import scipy


class Volume_to_Volume(t.nn.Module):
    """
    class to perform 3d-3d registration for initial alignment, fixed_image_volume is aligned to stack by inv_affine
    affine can be used later to resample stack to the fixed_volume
    """
    def __init__(self, device):
        super().__init__()

        self.device = device
        self.rotations = t.nn.ParameterList([t.nn.Parameter(t.zeros(3, device = self.device)) for i in range(1)])
        self.translations = t.nn.ParameterList([t.nn.Parameter(t.zeros(3, device = self.device)) for i in range(1)])
        self.affine_layer = monai.networks.layers.AffineTransform(mode = "bilinear",  normalized = True, padding_mode = "zeros")
        self.sav_gol_layer = monai.networks.layers.SavitzkyGolayFilter(7,3,axis=3,mode="zeros")

    
    def forward(self, fixed_volume_tensor:t.tensor, fixed_volume_meta:dict, stack_meta:dict, mode = "bilinear")->tuple:
        """
        fixed_volume tensor is transformed by current roation and translation parameters of the model (to be precise the inverse of their affine)
        the actual affine is returned to align the stack to the fixed image outside this module

        Args:
            fixed_volume_tensor (t.tensor): common volume registration target
            fixed_volume_meta (dict): meta of common volume
            stack_meta (dict): meta of current stac
            mode (str, optional): interpolation for resampling and spatial transform of fixed volume. Defaults to "bilinear".

        Returns:
            tuple: fixed volume after transform, affine for stack
        """
        resampler = monai.transforms.ResampleToMatch(mode = mode)
        
        #create affines and inv affines
        aff = self.create_T(self.rotations[0], self.translations[0])
        inv_aff = t.linalg.inv(aff)
        affines = aff.unsqueeze(0)
        inv_affines = inv_aff.unsqueeze(0)
        
        #prepare fixed_volume tensor, resmapl
        fixed_volume_tensor = fixed_volume_tensor.squeeze().unsqueeze(0)
        fixed_volume_tensor, fixed_volume_meta = resampler(fixed_volume_tensor,src_meta=fixed_volume_meta,
                         dst_meta=stack_meta)
        fixed_volume_meta["spatial_shape"] = np.array(list(fixed_volume_tensor.shape)[1:])
        fixed_volume_tensor_batch = fixed_volume_tensor.unsqueeze(0)
        

        fixed_volume_tensor_transformed = self.affine_layer(fixed_volume_tensor_batch, inv_affines)

        fixed_volume_tensor_transformed = fixed_volume_tensor_transformed.cpu()
        fixed_volume_tensor_transformed = self.sav_gol_layer(fixed_volume_tensor_transformed)
        fixed_volume_tensor_transformed = fixed_volume_tensor_transformed.to(self.device)

        return fixed_volume_tensor_transformed, affines
    
    def create_T(self,rotations:list, translations:list) -> t.tensor:
        """
        Creates affine matrix from rotations and translations

        Args:
            rotations (list): 
            translations (list):

        Returns:
            t.tensor: affine matrix
        """
        rotation_tensor = monai.transforms.utils.create_rotate(3, rotations, device = self.device,  backend="torch")
        translation_tensor = monai.transforms.utils.create_translate(3, translations, device = self.device, backend="torch")
        T = t.matmul(rotation_tensor,translation_tensor)
        return T
            

        
class Volume_to_Slice(t.nn.Module):
    """
    class to perform 3d-2d registration, aligns the fixed image to a slice by "inv_affines",
    affine can be used later to resample the slice to the fixed image
    """
    def __init__(self, n_slices:int, device, mode = "bilinear", tio_mode = "welch"):
        super().__init__()
        
        self.device = device
        self.n_slices = n_slices
        self.rotations = t.nn.ParameterList([t.nn.Parameter(t.zeros(3, device = self.device)) for i in range(n_slices)])
        self.translations = t.nn.ParameterList([t.nn.Parameter(t.zeros(3, device = self.device)) for i in range(n_slices)])
        self.affine_layer = monai.networks.layers.AffineTransform(mode = "bilinear",  normalized = True, padding_mode = "zeros")
        self.sav_gol_layer = monai.networks.layers.SavitzkyGolayFilter(7,3,axis=3,mode="zeros")
        self.mode = mode
        self.tio_mode = tio_mode

    def forward(self, fixed_image_tensor:t.tensor, fixed_image_affine:t.tensor, local_stack_tensor:t.tensor, local_stack_affine:t.tensor)->tuple:
        """
        fixed volume is transformed by current parameter (rotations, translations) (to be precise by inverse of their affine)
        the actual affines (one per slice) are returned to be applied outside this module

        Args:
            fixed_image_tensor (t.tensor): image_tensor of registration target/volume
            fixed_image_meta (dict): dictionary with meta data of fixed_image
            local_stack_tensor(t.tensor): image_tensor of local stack
            local_stack_meta (dict): dictionary with meta data of local stack
            mode (str, optional): interpolation mode for resampling


        Returns:
            tuple: tensor containing the fixed images transformed by the inverse affines of each slice, affines for slices
        """
        local_stack_tio = tio.Image(tensor=local_stack_tensor, affine = local_stack_affine)
        resampler_tio = tio.transforms.Resample(local_stack_tio, image_interpolation= self.tio_mode)
        resampler = monai.transforms.ResampleToMatch(mode = self.mode)
        add_channel = AddChanneld(keys=["image"])
        
        
        #resample fixed image to local stack and repeat n_slices time for batch-format
        """
        fixed_image_tensor = fixed_image_tensor.squeeze().unsqueeze(0)
        fixed_image_tensor, fixed_image_meta = resampler(fixed_image_tensor,src_meta=fixed_image_meta,
                         dst_meta=local_stack_meta)
        fixed_image_meta["spatial_shape"] = np.array(list(fixed_image_tensor.shape)[1:])
        
        fixed_image_tensor = fixed_image_tensor.unsqueeze(0)
        
        fixed_image_image_batch = fixed_image_tensor.repeat(self.n_slices,1,1,1,1)
        """
        
        fixed_tio = tio.Image(tensor=fixed_image_tensor.squeeze().unsqueeze(0).detach().cpu(), affine=fixed_image_affine) 
        fixed_tio = resampler_tio(fixed_tio)
        fixed_image_tensor = fixed_tio.tensor.to(self.device)
        fixed_image_image_batch = fixed_image_tensor.repeat(self.n_slices,1,1,1,1)


        #create affines and inv affines
        aff = self.homogenous_affine(self.rotations[0],self.translations[0])

        aff = self.homogenous_affine(self.rotations[0], self.translations[0])
        inv_aff = t.linalg.inv(aff)
        affines = aff.unsqueeze(0)
        inv_affines = inv_aff.unsqueeze(0)
        
        for sli in range(1,self.n_slices):
            aff = self.homogenous_affine(self.rotations[sli], self.translations[sli])
            inv_aff = t.linalg.inv(aff)
    
            affines = t.cat((affines,aff.unsqueeze(0)),0)
            inv_affines = t.cat((inv_affines,inv_aff.unsqueeze(0)),0)

        #slice simulation by transforming fixed image by inv affine --> layer are only 2 dimensional and align with the slices
        # note that fixed image was resamples to local stack so dimensionality matches and we can forward the transformed fixed images directly to the loss
        
        
        fixed_image_tran = self.affine_layer(fixed_image_image_batch, inv_affines)

        #SavGol filter requires tensor on cpu
        fixed_image_tran = fixed_image_tran.cpu()

        fixed_image_tran = self.sav_gol_layer(fixed_image_tran)

        fixed_image_tran = fixed_image_tran.to(self.device)
        

        #fixed_image_tran = t.mul(fixed_image_tensor,self.rotations[0][0])
        
        return fixed_image_tran, affines

    def create_T(self,rotations, translations):
        """
        Creates affine matrix from rotations and translations

        Args:
            rotations (list): 
            translations (list):

        Returns:
            t.tensor: affine matrix
        """
        #rotation = self.rotation_matrix(rotations).to(self.device)
        rotation_tensor = monai.transforms.utils.create_rotate(3, rotations, device = self.device,  backend="torch")
        translation_tensor = monai.transforms.utils.create_translate(3, translations, device = self.device, backend="torch")
        T = t.matmul(rotation_tensor,translation_tensor)
        return T
    
    def homogenous_affine(self,rotations:t.tensor,translations:t.tensor)->t.tensor:
        roll = rotations[0]
        yaw =rotations[1]
        pitch = rotations[2]

        tensor_0 = t.tensor(0.0,requires_grad=True)
        tensor_1 = t.tensor(1.0,requires_grad=True)

        RX = t.stack([
                        t.stack([tensor_1, tensor_0, tensor_0, tensor_0]),
                        t.stack([tensor_0, t.cos(roll), -t.sin(roll), tensor_0]),
                        t.stack([tensor_0, t.sin(roll), t.cos(roll), tensor_0]),
                        t.stack([tensor_0, tensor_0, tensor_0, tensor_1])]).reshape(4,4)

        RY = t.stack([
                        t.stack([t.cos(pitch), tensor_0, t.sin(pitch), tensor_0]),
                        t.stack([tensor_0, tensor_1, tensor_0, tensor_0]),
                        t.stack([-t.sin(pitch), tensor_0, t.cos(pitch), tensor_0]),
                        t.stack([tensor_0, tensor_0, tensor_0, tensor_1])]).reshape(4,4)

        RZ = t.stack([
                        t.stack([t.cos(yaw), -t.sin(yaw), tensor_0, translations[0]]),
                        t.stack([t.sin(yaw), t.cos(yaw), tensor_0, translations[1]]),
                        t.stack([tensor_0, tensor_0, tensor_1, translations[2] ]),
                        t.stack([tensor_0, tensor_0, tensor_0, tensor_1])]).reshape(4,4)

        R = t.mm(RZ, RY)
        R = t.mm(R, RX)

        return R

"""    
class PSF(t.nn.Module):
    def __init__(self, kernel_size) -> None:
        super().__init__()
        padding = np.floor(kernel_size/2)
        self.conv3d = t.nn.Conv3d(1,1, kernel_size, 1, padding, bias=False)
    
    def forward(self,x):
        return self.conv3d(x)

    def weights_init(self, kernel_size):
        kernel_template = np.zeros(kernel_size,kernel_size,kernel_size)
        kernel_center = np.floor(kernel_size)
        kernel_template[kernel_center,kernel_center,kernel_center] = 1 
        init_kernel = scipy.ndimage.gaussian_filter(kernel_template,sigma=1.5)
        for _, f in self.conv3d.named_parameters():
            f.data.copy_(t.from_numpy(init_kernel).unsqueeze(0).unsqueeze(0))
"""