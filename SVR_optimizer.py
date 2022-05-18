import torchio as tio
import monai
from monai.transforms import (
    AddChanneld,
    Affine
)
import torchvision as tv
import os
import numpy as np
import custom_models
import torch as t
from copy import deepcopy
import loss_module
import time
import matplotlib.pyplot as plt
from SVR_Preprocessor import Preprocesser
from SVR_outlier_removal import Outlier_Removal_Voxels, Outlier_Removal_Slices
from torch.utils.tensorboard import SummaryWriter

import SimpleITK as sitk

class SVR_optimizer():
    def __init__(self, src_folder:str, prep_folder:str, result_folder:str, stack_filenames:list, mask_filename:str, pixdims:list, device:str, monai_mode:str, tio_mode:str)->None:
        """
        constructer of SVR_optimizer class

        Args:
            src_folder (str): initial nifti_files
            prep_folder (str): folder to save prepocessed files
            result_folder (str): folder to save reconstruction results
            stack_filenames (list): of filenames of stacks to be reconstructed
            mask_filename (str): nifti filename to crop input images
            pixdims (list): ist of pixdims with increasing resolution
            device (str): _description_
            monai_mode (str): interpolation mode for monai resampling
            tio_mode (str): interpolation mode for monai resampling
        """
        timer = time.time()
        
        self.device = device
        print(f'Program runs on: {self.device}')
        self.stack_filenames = stack_filenames
        self.k = len(self.stack_filenames)
        self.mode = monai_mode
        self.pixdims = pixdims
        
        self.svr_preprocessor = Preprocesser(src_folder, prep_folder, result_folder, stack_filenames, mask_filename, device, monai_mode, tio_mode)
        
        self.fixed_image, self.stacks, self.slice_dimensions = self.svr_preprocessor.preprocess_stacks_and_common_vol(self.pixdims[0])
        
        self.ground_truth = self.stacks

        self.tio_mode = tio_mode

        self.writer = SummaryWriter("runs/eight_epochs_ncc")


          

    def construct_slices_from_stack(self, stack:dict, slice_dim):
        """Constructs slices from a single stack

        Args:
            stack (dict): stack that should be sliced

        Returns:
            slices: list of slices - each a 5d tensor
            n_slices: list of int: number of slice in that slice
            slice_dim: list of ints: dimension which is sliced
        """
        add_channel = AddChanneld(keys=["image"])
        stack = add_channel(stack)
        stack_image = stack["image"]

        n_slices =  min(list(stack_image.shape[2:]))
        
        slices = t.zeros_like(stack_image).repeat(n_slices,1,1,1,1)
        
        if slice_dim == 0:
            for i in range (0,n_slices):
                tmp = deepcopy(stack_image)
                tmp[:,:,:i,:,:] = 0
                tmp[:,:,i+1:,:,:] = 0
                slices[i,:,:,:,:] = tmp
        elif slice_dim == 1:
            for i in range (0,n_slices):
                tmp = deepcopy(stack_image)
                tmp[:,:,:,:i,:] = 0
                tmp[:,:,:,i+1:,:] = 0
                slices[i,:,:,:,:] = tmp
        elif slice_dim == 2:
            for i in range (0,n_slices):
                tmp = deepcopy(stack_image)
                tmp[:,:,:,:,:i] = 0
                tmp[:,:,:,:,i+1:] = 0
                slices[i,:,:,:,:] = tmp
        return slices, n_slices



    def optimize_volume_to_slice(self, epochs:int, inner_epochs:int, lr, loss_fnc = "ncc", opt_alg = "Adam"):
        """
        optimizes transform of individual slices to mitigate motion artefact, uses initial 3d-3d registration
        implemented in SVR_Preprocessor

        Args:
            epochs (int): epochs of registration of all stacks
            inner_epochs (int): epochs of 3d-2d registration of each stack
            lr (_type_): learning rate of optimizer
            loss_fnc (str, optional): loss function Defaults to "ncc".
            opt_alg (str, optional): optimization algorithm Defaults to "Adam".
        """
        models = list()
        slices = list()
        n_slices = list()
        slice_dims = self.slice_dimensions
        affines_slices = list()
        
        #Afffine transformations for updating common volume from slices (use bilinear because it's 2d transform)
        affine_transform_slices = monai.networks.layers.AffineTransform(mode = "bilinear",  normalized = True, padding_mode = "zeros")
        
        for st in range(0,self.k):
            slice_tmp, n_slice = self.construct_slices_from_stack(self.stacks[st], slice_dims[st])
            slices.append(slice_tmp)
            n_slices.append(n_slice)
            #models.append(custom_models.Volume_to_Slice(n_slices=n_slice, device=self.device))
            #store affine transforms
            affines_slices.append(t.eye(4, device=self.device).unsqueeze(0).repeat(n_slice,1,1))
            
                          
        loss = loss_module.Loss_Volume_to_Slice(loss_fnc, self.device)
        resampler = monai.transforms.ResampleToMatch(mode = self.mode)
        
        common_volume = t.zeros_like(self.fixed_image["image"], device=self.device)
        fixed_image_tensor = self.fixed_image["image"]
        fixed_image_meta = self.fixed_image["image_meta_dict"]
        
        #use this template for tio-resampling operations of stacks during update
        tio_fixed_image_template = self.svr_preprocessor.monai_to_torchio(self.fixed_image)
        resampling_to_fixed_tio = tio.transforms.Resample(tio_fixed_image_template, image_interpolation=self.tio_mode)

        for epoch in range(0,epochs):
            likelihood_image_storage = list()
            tio_fixed_image_template = self.svr_preprocessor.monai_to_torchio({"image": fixed_image_tensor, "image_meta_dict": fixed_image_meta})
            resampling_to_fixed_tio = tio.transforms.Resample(tio_fixed_image_template, image_interpolation=self.tio_mode)
            print(f'\n\n Epoch: {epoch}')

            for st in range (0, self.k):
                print(f"\n  stack: {st}")
                model = custom_models.Volume_to_Slice(n_slices=n_slices[st], device=self.device, mode = self.mode, tio_mode = self.tio_mode)
                model.to(self.device)

                
                
                if opt_alg == "SGD":
                    optimizer = t.optim.SGD(model.parameters(), lr = lr)
                elif(opt_alg == "Adam"):
                    optimizer = t.optim.Adam(model.parameters(), lr = lr)
                else:
                    assert("Choose SGD or Adam as optimizer")
                
                local_stack = self.stacks[st]
                local_slices = slices[st]

                local_stack_tio = self.svr_preprocessor.monai_to_torchio(local_stack)
                
                #optimization procedure
                for inner_epoch in range(0,inner_epochs):
                    model.train()
                    optimizer.zero_grad()
                    #return fixed_images resamples to local stack where inverse affines were applied
                    #in shape (n_slices,1,[stack_shape]) affines 
                    
                    tr_fixed_images, affines_tmp = model(fixed_image_tensor.detach(), fixed_image_meta["affine"], local_stack_tio.tensor, local_stack_tio.affine)
                    """
                    if epoch == 0 and st == 0:
                        model_tensor_board = custom_models.Volume_to_Slice(n_slices=2, device=self.device, mode = self.mode, tio_mode = self.tio_mode)
                        red_input = fixed_image_tensor[:,:,0:2,:,:].detach()
                        self.writer.add_graph(model_tensor_board,(red_input, t.tensor(fixed_image_meta["affine"]), local_stack_tio.tensor, t.tensor(local_stack_tio.affine)))
                        self.writer.close
                    """
                    tr_fixed_images = tr_fixed_images.to(self.device)
                    
                    #calcuates 2d between a local slice and the corresponding slice in the tr_fixed_image
                    loss_tensor = loss(tr_fixed_images, local_slices, n_slices[st], slice_dims[st])
                    print(f'loss: {loss_tensor.item()}')
                    loss_tensor.backward(retain_graph = False)

                    if inner_epoch == 0:
                        self.writer.add_scalar(f"Loss_stack_{st}", loss_tensor.item(), epoch)
                    #torchviz.make_dot(loss_tensor, params= dict(model.named_parameters()))
                    """
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            print (f'parameter {name} has only zero values: {t.all(param.data == 0)}')
                    """
                    
                    optimizer.step()
                
                self.writer.close()
            #update procedure
                timer = time.time()
                for sl in range(0,n_slices[st]):
                    #multipy the new transform to the existing transform
                    #order second argument is the first transform
                    #this is necessary because the reference image was moved by by affines_tmp
                    affines_slices[st][sl,:,:] = t.matmul(affines_tmp[sl],affines_slices[st][sl,:,:])
                
                timer = time.time()
                likelihood_images_voxels = self.outlier_removal_voxels(tr_fixed_images,local_slices, n_slices[st], slice_dims[st])
                likelihood_images_slices = self.outlier_removal_slices(likelihood_images_voxels, n_slices[st], slice_dims[st])
                print(f'outlier removal:  {time.time() - timer} s ')

                timer = time.time()
                #multiply likelihood to each voxel for outlier removal
                #local_slices = t.mul(local_slices, t.mul(likelihood_images_voxels, likelihood_images_slices))
                local_slices = t.mul(local_slices, likelihood_images_voxels)

                affines_tmp = affines_slices[st]
                #apply affines to transform slices
                #the layer can only use "bilinear"

                #comment out for benchmark
                # use identitiy for benchmark
                #affines_tmp = t.eye(4).repeat(n_slices[st],1,1)


                transformed_slices = affine_transform_slices(local_slices, affines_tmp)

                """
                #try Affine transform - should be able to use "bicubic"
                transformed_slices = t.zeros_like(local_slices)
                
                for sl in range(0,n_slices[st]):
                    tio_image = tio.Image(tensor = transformed_slices[sl,:,:,:,:].detach().cpu(), affine = local_stack_tio.affine)
                    transformed_slices[sl,:,:,:,:] = self.sitk_affine_transform(tio_image,affines_tmp[sl])
                """

                #leaves out 3d 2d registration
                #transformed_slices = local_slices

                transformed_slices = transformed_slices.detach()
                
                #update current stack from slices
                
                common_stack = t.zeros_like(common_volume, device=self.device)

                for sl in range(0,n_slices[st]):
                    tmp = transformed_slices[sl,:,:,:,:]
                    tmp_tio = tio.Image(tensor=tmp.squeeze().unsqueeze(0).detach().cpu(), affine=local_stack["image_meta_dict"]["affine"])
                    tio_transformed = resampling_to_fixed_tio(tmp_tio)
                    common_stack = common_stack + tio_transformed.tensor.unsqueeze(0).to(self.device)
                print(f'common vol update:  {time.time() - timer} s ')
                #update common volume from stack
                common_volume = common_volume + common_stack
            
            normalizer = tv.transforms.Normalize(t.mean(common_volume), t.std(common_volume))
            common_volume = normalizer(common_volume)

            timer = time.time()
            fixed_image_tensor = common_volume
            if epoch < epochs - 1:
                upsample_bool = (self.pixdims[epoch + 1] == self.pixdims[epoch])
                fixed_image = self.svr_preprocessor.save_intermediate_reconstruction_and_upsample(fixed_image_tensor, fixed_image_meta, epoch, upsample=upsample_bool, pix_dim = self.pixdims[epoch+1])
                fixed_image_tensor = fixed_image["image"]
                fixed_image_meta = fixed_image["image_meta_dict"]
                common_volume = t.zeros_like(fixed_image_tensor)
            else:
                self.svr_preprocessor.save_intermediate_reconstruction(fixed_image_tensor,fixed_image_meta,epoch)
            print(f'fixed_volume update:  {time.time() - timer} s ')


    def outlier_removal_voxels(self, tr_fixed_images:t.tensor, local_slices:t.tensor, n_slices:int, slice_dim:int) -> t.tensor:
        likelihood_images = t.zeros_like(local_slices, device=self.device)
        for sl in range (0,n_slices):
            #outlier removal
            if slice_dim == 0:
                pred = tr_fixed_images[sl,0,sl,:,:]
                target = local_slices[sl,0,sl,:,:]
            elif slice_dim == 1:
                pred = tr_fixed_images[sl,0,:,sl,:]
                target = local_slices[sl,0,:,sl,:]
            elif slice_dim == 2:
                pred = tr_fixed_images[sl,0,:,:,sl]
                target = local_slices[sl,0,:,:,sl]

            error_tensor = pred - target

            outlier_remover = Outlier_Removal_Voxels(error_tensor)
            p_voxels = outlier_remover(error_tensor)

            if slice_dim == 0:
                likelihood_images[sl,0,sl,:,:] = p_voxels
            elif slice_dim == 1:
                likelihood_images[sl,0,:,sl,:] = p_voxels
            elif slice_dim == 2:
                likelihood_images[sl,0,:,:,sl] = p_voxels
        return likelihood_images

    def outlier_removal_slices(self,likelihood_images:t.tensor, n_slices:int, slice_dim:int)->t.tensor:

        outlier_remover = Outlier_Removal_Slices()

        likelihood_images_squared = t.pow(likelihood_images,t.tensor(2))
        if slice_dim == 0:
            voxels_per_slice = t.numel(likelihood_images_squared[0,0,0,:,:])
            red_voxel_prob = t.sqrt( t.einsum('ijklm->k',likelihood_images_squared) / voxels_per_slice )
        elif slice_dim == 1:
            voxels_per_slice = t.numel(likelihood_images_squared[0,0,:,0,:])
            red_voxel_prob = t.sqrt( t.einsum('ijklm->l',likelihood_images_squared) / voxels_per_slice )
        elif slice_dim == 2:
            voxels_per_slice = t.numel(likelihood_images_squared[0,0,:,:,0])
            red_voxel_prob = t.sqrt( t.einsum('ijklm->m',likelihood_images_squared) / voxels_per_slice )
        
        p_slices = outlier_remover(red_voxel_prob)
        p_slices_output = t.zeros_like(likelihood_images)
        if slice_dim == 0:
            for i in range(0,n_slices):
                p_slices_output[i,0,i,:,:] = p_slices[i]
        elif slice_dim == 1:
            for i in range(0,n_slices):
                p_slices_output[i,0,:,i,:] = p_slices[i]
        elif slice_dim == 2:
            for i in range(0,n_slices):
                p_slices_output[i,0,:,:,i] = p_slices[i]
        
        return p_slices_output

    def sitk_affine_transform(self, tio_image:tio.Image, affine_matr:t.tensor)->t.tensor:
        sitk_image = tio_image.as_sitk()

        rotation = affine_matr[:3,:3].ravel().tolist()
        translation = affine_matr[:3,3].tolist()
        affine = sitk.AffineTransform(rotation,translation)

        reference_image = sitk_image
        interpolator = sitk.sitkWelchWindowedSinc
        default_value = 0

        resampled =  sitk.Resample(sitk_image,reference_image,affine,interpolator,default_value)

        tensor = t.permute(t.tensor(sitk.GetArrayFromImage(resampled)),(2,1,0))

        tensor = tensor.unsqueeze(0)
        return tensor




    
"""
               for name, param in model.named_parameters():
                if param.requires_grad:
                    print (f'parameter {name} has value not equal to zero: {t.all(param.data == 0)}')


            timer = time.time()
                    optimizer.step()
                    print(f'optimizer:  {time.time() - timer} s ')
"""
