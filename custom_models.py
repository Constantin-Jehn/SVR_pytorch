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
    
    def forward(self, im_slices):
        for sli in range(0,self.n_slices):
            im_slices[sli] = self.affine_layer(im_slices[sli], self.create_T(self.rotations[sli],self.translations[sli]))
        return im_slices  
    
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
            
            
class ResamplingToFixed(t.nn.Module):
    def __initi__(self):
        super().__init__()
        
    def forward(self, im_slices, local_stack, fixed_image):
        add_channel = AddChanneld(keys=["image"])
        local_stack = add_channel(local_stack)
        
        local_stack = self.update_local_stack(im_slices, local_stack)
        
        local_stack["image"] = t.squeeze(local_stack["image"]).unsqueeze(0)
        
        local_stack = self.resample_to_fixed_image(local_stack, fixed_image)
        
        local_stack = add_channel(local_stack)
        
        return local_stack
        
        
        
    def update_local_stack(self, im_slices, local_stack):
        """
        use to update the stacks after transform (which should be optimized)
        was applied to slices
        Parameters
        ----------
        im_slices : list 
            contains slices of one stack
        local_stack : dict
            local_stack to be updated
        Returns
        -------
        None
        """
        n_slices = len(im_slices)
        local_stack["image"] = im_slices[0]
        for sli  in range(1,n_slices):
            local_stack["image"] = local_stack["image"] + im_slices[sli]
        #update target_dict
        return local_stack
    
    
    def resample_to_fixed_image(self, local_stack, fixed_image):
        """
        resamples the updated stack in "self.ground_truth" into "self.stacks"
        for loss computation
        Parameters
        ----------
        local_stack : dict
            updated stack on local coordinates
        fixed_image : dict
            stck in world coordinates (fixed_image)

        Returns
        -------
        world_stack:
            local_stack in world coordinates

        """
        dst_meta = fixed_image["image_meta_dict"]
        resampler = monai.transforms.ResampleToMatch()
        
        world_stack = {"image": t.zeros(1), "image_meta_dict":{}}
        
        file_obj = deepcopy(local_stack["image_meta_dict"]["filename_or_obj"])
        original_affine = deepcopy(local_stack["image_meta_dict"]["affine"])
        world_stack["image"], world_stack["image_meta_dict"] = resampler(local_stack["image"],src_meta = local_stack["image_meta_dict"], 
                                                                                      dst_meta = dst_meta, padding_mode = "zeros")
        
        world_stack["image_meta_dict"]["filename_or_obj"] = file_obj
        world_stack["image_meta_dict"]["original_affine"] = original_affine 
        
        return world_stack
        
        