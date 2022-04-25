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
import os
import numpy as np
import custom_models
import torch as t
from copy import deepcopy
import loss_module
import time
import matplotlib.pyplot as plt
from torchviz import make_dot
from SVR_Preprocessor import Preprocesser


class SVR_optimizer():
    def __init__(self, src_folder, prep_folder, stack_filenames, mask_filename, pixdims, device, mode):
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
        
        svr_preprocessor = Preprocesser(src_folder, prep_folder, stack_filenames, mask_filename, pixdims, device, mode)
        
        self.fixed_images, self.stacks = svr_preprocessor.preprocess_stacks_and_common_vol()
        
        self.ground_truth = svr_preprocessor.get_cropped_stacks()
        
        print(f'preprocessing done in {time.time() - timer} s')
        
        """
        add_channel = AddChanneld(keys=["image"])
        to_device = monai.transforms.ToDeviced(keys = ["image"], device = self.device)
        
        self.fixed_images = self.create_common_volume_registation()
        self.fixed_images = add_channel(self.fixed_images)
        
        #self.fixed_images = self.create_multiresolution_fixed_images(pixdims)
        # for i in range(0,len(self.fixed_images)):
        #     self.fixed_images[i] = add_channel(self.fixed_images[i])
        #     self.fixed_images[i]["image"].requires_grad = False
        #     self.fixed_images[i] = to_device(self.fixed_images[i])
       
        
        
        # self.initial_vol = {"image": t.zeros_like(self.fixed_images[0]["image"]), "image_meta_dict": deepcopy(self.fixed_images[0]["image_meta_dict"])}
        # self.initial_vol = to_device(self.initial_vol)
        print("fixed_images_generated")
        
        
        self.crop_images(upsampling = False)
        #remains in initial coordiate system
        self.ground_truth = self.load_stacks(to_device=True)
        
        #self.ground_truth, self.im_slices, self.target_dict, self.k = self.preprocess()

        """    
    
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
    
    def optimize_multiple_stacks(self, epochs:int, inner_epochs:int, lr, multi_res = False,  loss_fnc = "ncc", opt_alg = "Adam"):
        """
        Parameters
        ----------
        epochs : integer
            epochs of 2D/3D registration per stack
        lr : float
            optimization hyperparameter
        loss_fnc : string, optional
             The default is "ncc".
        opt_alg : string, optional
            DESCRIPTION. The default is "Adam".

        Returns
        -------
        world_stack : dict
            reconstruncted volume
        loss_log : list
            list of losses

        """
        models = list()
        #create model for each stack 
        slices = list()
        slice_dims = list()
        for st in range (0,self.k):
            slice_tmp, n_slices, slice_dim = self.construct_slices_from_stack(self.stacks[st])
            slices.append(slice_tmp)
            slice_dims.append(slice_dim)
            #n_slices = self.ground_truth[st]["image"].shape[-1]
            models.append(custom_models.Reconstruction(n_slices = n_slices, device = self.device))
        
        #resampling_model = custom_models.ResamplingToFixed()
        #resampling_model.to(self.device)
        
        loss = loss_module.RegistrationLossSingleSlice(loss_fnc, self.device)
    
        loss_log = np.zeros((epochs,self.k,inner_epochs))
        
        resampler = monai.transforms.ResampleToMatch(mode = self.mode)
        
        if multi_res:
            common_volume = t.zeros_like(self.fixed_images[0]["image"])
        else:
            common_volume = t.zeros_like(self.fixed_images["image"])
            fixed_image = self.fixed_images
            
        for epoch in range(0,epochs):
            if multi_res:
                fixed_image = self.fixed_images[epoch]
                if epoch > 0:
                    common_volume = common_volume.squeeze().unsqueeze(0)
                    fixed_image["image"],_ = resampler(common_volume,src_meta=self.fixed_images[epoch - 1]["image_meta_dict"],
                                                     dst_meta = fixed_image["image_meta_dict"])
                    
                    fixed_image["image"] = fixed_image["image"].unsqueeze(0)
                    common_volume = t.zeros_like(fixed_image["image"])
            else:
                if epoch > 0:
                    fixed_image["image"] = common_volume
                    common_volume = t.zeros_like(self.fixed_images["image"])
                
            
            #self.initial_vol["image"] = t.zeros_like(self.fixed_image["image"])

            print(f'\n\n Epoch: {epoch}')
            
            #plt.imshow(self.fixed_image["image"][0,0,:,:,20].detach().numpy())
            #plt.show()
            
            for st in range(0,self.k):
                #loss_stack = list()
                print(f"\n  stack: {st}")
                model = models[st]
                model.to(self.device)
                
                #in batch first shape
                slices_tmp = slices[st]
                
                if opt_alg == "SGD":
                    optimizer = t.optim.SGD(model.parameters(), lr = lr)
                elif(opt_alg == "Adam"):
                    optimizer = t.optim.Adam(model.parameters(), lr = lr)
                else:
                    assert("Choose SGD or Adam as optimizer")

                local_stack = self.ground_truth[st]
                
                for inner_epoch in range(0,inner_epochs):
                    model.train()
                    optimizer.zero_grad()
                    timer = time.time()
                    transformed_slices = model(slices_tmp.detach(), local_stack["image_meta_dict"], fixed_image["image_meta_dict"], transform_to_fixed = False, mode = self.mode)
                    transformed_slices = transformed_slices.to(self.device)
                    print(f'forward pass. {time.time() - timer} s ')
                    
                    dot = make_dot(transformed_slices[0,:,:,:,:], params = dict(model.named_parameters()))
                    
                    timer = time.time()
                    loss_tensor = loss(transformed_slices, fixed_image, slice_dims[st])
                    
                    print(f'loss:  {time.time() - timer} s ')
                    
                    print(f'Epoch: {epoch} loss: {loss_tensor.item()}')
                    timer = time.time()
                    
                    loss_tensor.backward(retain_graph = False)
                    
                    print(f'backward:  {time.time() - timer} s ')
                    print(f'loss: {loss_tensor.item()}')
                    loss_log[epoch,st,inner_epoch] = loss_tensor
                    timer = time.time()
                    optimizer.step()
                    print(f'optimizer:  {time.time() - timer} s ')

                #update common_volume
                transformed_slices = transformed_slices.detach()
                
                
                #if loss was applied in hr
                common_volume = common_volume + t.sum(transformed_slices, dim = 0).unsqueeze(0)
                
                # for sl in range(0,transformed_slices.shape[0]):
                #     sli = transformed_slices[sl,:,:,:,:]
                #     resampled, _ = resampler(sli, src_meta=self.ground_truth[st]["image_meta_dict"], dst_meta = self.fixed_image["image_meta_dict"])
                #     self.initial_vol["image"] = self.initial_vol["image"] + resampled.unsqueeze(0)

                print('inital_updated')
                
                
            common_volume = t.div(common_volume, self.k)
            
            #self.fixed_image["image"] = self.initial_vol["image"]
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print (f'parameter {name} has value not equal to zero: {t.all(param.data == 0)}')
            
        #cast to image format
        fixed_image["image"] = common_volume
        fixed_image["image"] = t.squeeze(fixed_image["image"]).unsqueeze(0)
        
        #loss_log = 0
        
        return fixed_image, loss_log

    def bending_loss_fucntion_single_stack(target_dict_image):
        monai_bending = monai.losses.BendingEnergyLoss()
        return monai_bending(target_dict_image.expand(-1,3,-1,-1,-1))