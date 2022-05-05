import torchio as tio
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
from SVR_outlier_removal import outlier_removal


class SVR_optimizer():
    def __init__(self, src_folder, prep_folder, result_folder, stack_filenames, mask_filename, pixdims, device, mode):
        """
        constructer of SVR_optimizer class
        Parameters
        ----------
        src_folder : string
            initial nifti_files
        prep_folder : string
            folder to save prepocessed files
        stack_filenames : list
            of filenames of stacks to be reconstructed
        mask_filename : string
            nifti filename to crop input images
        pixdim : list
            list of pixdims with increasing resolution
        device : TYPE
            DESCRIPTION.
        mode : string
            interpolation mode

        Returns
        -------
        None.

        """
        timer = time.time()
        
        self.device = device
        print(f'Program runs on: {self.device}')
        self.stack_filenames = stack_filenames
        self.k = len(self.stack_filenames)
        self.mode = mode

        self.pixdims = pixdims
        
        self.svr_preprocessor = Preprocesser(src_folder, prep_folder, result_folder, stack_filenames, mask_filename, device, mode)
        
        self.fixed_images, self.stacks = self.svr_preprocessor.preprocess_stacks_and_common_vol(self.pixdims[0])
        
        self.ground_truth = self.stacks

        self.tio_mode = "gaussian"
          
    
    def create_common_volume(self):
        """
        Combine updated local stacks that are in common coordinate system, to 
        one superpositioned volume

        Returns
        -------
        world_stack : dictionary
            contains nifti file of the the reconstructed brain.

        """
        resampler = monai.transforms.ResampleToMatch(mode = self.mode)
        
        stacks = self.load_stacks()
        to_device = monai.transforms.ToDeviced(keys = ["image"], device = self.device)
        tmp = stacks[0]["image"]
        
        
        for st in range(1,self.k):
            image,_ = resampler(stacks[st]["image"], src_meta=stacks[st]["image_meta_dict"], 
                              dst_meta=stacks[0]["image_meta_dict"])
            tmp = tmp + image
        tmp = tmp/self.k
        world_stack = {"image":tmp, "image_meta_dict": stacks[0]["image_meta_dict"]}
        world_stack = to_device(world_stack)
        
        return world_stack
    
    def create_multiresolution_fixed_images(self, pixdims):
        """
        Parameters
        ----------
        pixdims : list
            different resolution where first is coarsest

        Returns
        -------
        fixed_images : list
            contains initial fixed images, only 0th will be used in the optimization
            the rest will be used as template for each resolution

        """
        n_pixdims = len(pixdims)
        fixed_images = list()
        
        for i in range(0,n_pixdims):
            self.crop_images(upsampling=True, pixdim = pixdims[i])
            fixed_image = self.create_common_volume_registration()
            fixed_images.append(fixed_image)
            
        return fixed_images

    def construct_slices_from_stack(self, stack):
        """
        Constructs slices from a single stack

        Parameters
        ----------
        stack : dict
            stack that should be sliced

        Returns
        -------
        slices : list
            list of slices - each a 5d tensor

        """
        add_channel = AddChanneld(keys=["image"])
        stack = add_channel(stack)
        stack_image = stack["image"]

        slice_dim, n_slices = list(stack_image.shape[2:]).index(min(list(stack_image.shape[2:]))),  min(list(stack_image.shape[2:]))
        
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
        return slices, n_slices, slice_dim



    def optimize_volume_to_slice(self, epochs:int, inner_epochs:int, lr, loss_fnc = "ncc", opt_alg = "Adam"):
        models = list()
        slices = list()
        n_slices = list()
        slice_dims = list()
        affines_slices = list()
        
        #Afffine transformations for updating common volume from slices (use bilinear because it's 2d transform)
        affine_transform_slices = monai.networks.layers.AffineTransform(mode = "bilinear",  normalized = True, padding_mode = "zeros")
        resampler_slices = monai.transforms.ResampleToMatch(mode = self.mode)
        
        for st in range(0,self.k):
            slice_tmp, n_slice, slice_dim = self.construct_slices_from_stack(self.stacks[st])
            slices.append(slice_tmp)
            n_slices.append(n_slice)
            slice_dims.append(slice_dim)
            models.append(custom_models.Volume_to_Slice(n_slices=n_slice, device=self.device))
            #store affine transforms
            affines_slices.append(t.eye(4, device=self.device).unsqueeze(0).repeat(n_slice,1,1))
            
                          
        loss = loss_module.Loss_Volume_to_Slice(loss_fnc, self.device)
        resampler = monai.transforms.ResampleToMatch(mode = self.mode)
        
        common_volume = t.zeros_like(self.fixed_images["image"])
        fixed_image_image = self.fixed_images["image"]
        fixed_image_meta = self.fixed_images["image_meta_dict"]
        
        #use this template for tio-resampling operations of stacks during update
        tio_fixed_image_template = self.svr_preprocessor.monai_to_torchio(self.fixed_images)
        resampling_to_fixed_tio = tio.transforms.Resample(tio_fixed_image_template, image_interpolation=self.tio_mode)

        for epoch in range(0,epochs):
            print(f'\n\n Epoch: {epoch}')

            for st in range (0, self.k):
                print(f"\n  stack: {st}")
                model = models[st]
                model.to(self.device)
                
                if opt_alg == "SGD":
                    optimizer = t.optim.SGD(model.parameters(), lr = lr)
                elif(opt_alg == "Adam"):
                    optimizer = t.optim.Adam(model.parameters(), lr = lr)
                else:
                    assert("Choose SGD or Adam as optimizer")
                
                local_stack = self.stacks[st]
                local_slices = slices[st]
                
                for inner_epoch in range(0,inner_epochs):
                    model.train()
                    optimizer.zero_grad()
                    #return fixed_images resamples to local stack where inverse affines were applied
                    #in shape (n_slices,1,[stack_shape]) affines 
                    tr_fixed_images, affines_tmp = model(fixed_image_image.detach(), fixed_image_meta, local_stack["image_meta_dict"], mode = self.mode)
                    
                    tr_fixed_images = tr_fixed_images.to(self.device)
                    
                    #calcuates 2d between a local slice and the corresponding slice in the tr_fixed_image
                    loss_tensor = loss(tr_fixed_images, local_slices, n_slices[st], slice_dims[st])
                    print(f'loss: {loss_tensor.item()}')
                    loss_tensor.backward(retain_graph = False)
                    
                    optimizer.step()
                
                likelihood_images = t.ones_like(local_slices, device=self.device)
                for sl in range(0,n_slices[st]):
                    #multithe new transform to the existing transform
                    #order second argument is the first transform

                    #this is necessary because the reference image was moved by by affines_tmp
                    affines_slices[st][sl,:,:] = t.matmul(affines_tmp[sl],affines_slices[st][sl,:,:])

                    slice_dim = slice_dims[st]
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

                    outlier_remover = outlier_removal(error_tensor)
                    p = outlier_remover(error_tensor)

                    if slice_dim == 0:
                        likelihood_images[sl,0,sl,:,:] = p
                    elif slice_dim == 1:
                        likelihood_images[sl,0,:,sl,:] = p
                    elif slice_dim == 2:
                        likelihood_images[sl,0,:,:,sl] = p


                #multiply likelihood to each voxel for outlier removal
                
                #local_slices = t.mul(local_slices,likelihood_images)

                affines_tmp = affines_slices[st]
                #apply affines to transform slices
                transformed_slices = affine_transform_slices(local_slices, affines_tmp)
                transformed_slices = transformed_slices.detach()
                
                common_stack = t.zeros_like(common_volume)
                for sl in range(0,n_slices[st]):
                    tmp = transformed_slices[sl,:,:,:,:]
                    """ 
                    can be replaced by torchio Gaussian interpolation
                    """
                    tmp_tio = tio.Image(tensor=tmp.squeeze().unsqueeze(0).detach().cpu(), affine=local_stack["image_meta_dict"]["affine"])
                    tio_transformed = resampling_to_fixed_tio(tmp_tio)
                    common_stack = common_stack + tio_transformed.tensor.unsqueeze(0)


                #common_volume = common_volume + t.div(common_stack, t.max(common_stack)/2047)
                common_volume = common_volume + common_stack
            
            #common_volume = t.div(common_volume,self.k)
            normalizer = tv.transforms.Normalize(t.mean(common_volume), t.std(common_volume))
            common_volume = normalizer(common_volume)

            #common_volume = t.div(common_volume, t.max(common_volume)/2047)
            # common_volume = self.svr_preprocessor.normalize(common_volume.squeeze().unsqueeze(0))
            # common_volume = common_volume.unsqueeze(0)
            #normalizer = tv.transforms.Normalize(t.mean(common_volume), t.std(common_volume))
            #common_volume = normalizer(common_volume)

            fixed_image_image = common_volume
            if epoch < epochs - 1:
                fixed_image = self.svr_preprocessor.save_intermediate_reconstruction_and_upsample(fixed_image_image, fixed_image_meta, epoch, self.pixdims[epoch+1])
                fixed_image_image = fixed_image["image"]
                fixed_image_meta = fixed_image["image_meta_dict"]
                common_volume = t.zeros_like(fixed_image_image)
            else:
                self.svr_preprocessor.save_intermediate_reconstruction(fixed_image_image,fixed_image_meta,epoch)
     
        #world_stack = {"image": common_volume.squeeze().unsqueeze(0), "image_meta_dict": self.fixed_images["image_meta_dict"]}
        #fixed_image["image"] = t.squeeze(fixed_image["image"]).unsqueeze(0)
        #loss_log = 0
        #return world_stack, loss_log
    
    def bending_loss_fucntion_single_stack(target_dict_image):
        monai_bending = monai.losses.BendingEnergyLoss()
        return monai_bending(target_dict_image.expand(-1,3,-1,-1,-1))

"""
               for name, param in model.named_parameters():
                if param.requires_grad:
                    print (f'parameter {name} has value not equal to zero: {t.all(param.data == 0)}')


            timer = time.time()
                    optimizer.step()
                    print(f'optimizer:  {time.time() - timer} s ')
"""