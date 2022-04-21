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


class SVR_optimizer():
    def __init__(self,src_folder, prep_folder, stack_filenames, mask_filename, pixdims, device, mode):
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
        self.src_folder = src_folder
        self.prep_folder = prep_folder
        self.stack_filenames = stack_filenames
        
        self.k = len(self.stack_filenames)
        self.mask_filename = mask_filename
        self.mode = mode

        
        self.fixed_images = self.create_multiresolution_fixed_images(pixdims)
        add_channel = AddChanneld(keys=["image"])
        to_device = monai.transforms.ToDeviced(keys = ["image"], device = self.device)
        
        for i in range(0,len(self.fixed_images)):
            self.fixed_images[i] = add_channel(self.fixed_images[i])
            self.fixed_images[i]["image"].requires_grad = False
            self.fixed_images[i] = to_device(self.fixed_images[i])
        print("fixed_images_generated")
        
        
        self.initial_vol = {"image": t.zeros_like(self.fixed_images[0]["image"]), "image_meta_dict": deepcopy(self.fixed_images[0]["image_meta_dict"])}
        self.initial_vol = to_device(self.initial_vol)
        
        self.crop_images(upsampling = False)
        #remains in initial coordiate system
        self.ground_truth = self.load_stacks(to_device=True)
        print(f'preprocessing done in {time.time() - timer} s')
        #self.ground_truth, self.im_slices, self.target_dict, self.k = self.preprocess()


    def crop_images(self, upsampling = False, pixdim = 0):
        """
        Parameters
        ----------
        src_folder : string
            folder of images to be cropped
        filenames : string
            stacks that should be cropped
        mask_filename : string
            mask for cropping
        dst_folder : string
            folder to save cropped files
        upsampling-flag:
            map all stacks in high resolution to common coord for creation of 
            initial hr fixed_image

        Returns
        -------
        None.
        """
        path_mask = os.path.join(self.src_folder, self.mask_filename)
        mask = tio.LabelMap(path_mask)
        path_dst = os.path.join(self.prep_folder, self.mask_filename)
        mask.save(path_dst)
        
        for i in range(0,self.k):
            filename = self.stack_filenames[i]
            path_stack = os.path.join(self.src_folder, filename)
            stack = tio.ScalarImage(path_stack)
            resampler = tio.transforms.Resample(stack)
            resampled_mask = resampler(deepcopy(mask))
            subject = tio.Subject(stack = stack, mask = resampled_mask)
            
            masked_indices_1 = t.nonzero(subject.mask.data)
            min_indices = np.array([t.min(masked_indices_1[:,1]).item(), t.min(masked_indices_1[:,2]).item(),t.min(masked_indices_1[:,3]).item()])
            max_indices = np.array([t.max(masked_indices_1[:,1]).item(), t.max(masked_indices_1[:,2]).item(),t.max(masked_indices_1[:,3]).item()])
            roi_size = (max_indices - min_indices) 
            
            cropper = tio.CropOrPad(list(roi_size),mask_name= 'mask')
            
            cropped_stack = cropper(subject)
            
            if upsampling:
                if i == 0:
                    #only upsample first stack, for remaining stack it's done by resamplich to this stack
                    upsampler = tio.transforms.Resample(pixdim)
                    cropped_stack = upsampler(cropped_stack)
                else:
                    path_stack = os.path.join(self.prep_folder, self.stack_filenames[0])
                    resampler = tio.transforms.Resample(path_stack)
                    cropped_stack = resampler(cropped_stack)

            path_dst = os.path.join(self.prep_folder, filename)
            cropped_stack.stack.save(path_dst)
            
    
    
    def load_stacks(self, to_device = False):
        """
        After cropping the initial images in low resolution are saved in their original coordinates
        for the loss computation
        Returns
        -------
        ground_truth : list
            list of dictionaries containing the ground truths

        """
        add_channel = AddChanneld(keys=["image"])
        loader = LoadImaged(keys = ["image"])
        to_tensor = ToTensord(keys = ["image"])
        if to_device:
            to_device = monai.transforms.ToDeviced(keys = ["image"], device = self.device)
        
        stack_list = list()
        for i in range(0,self.k):
            path = os.path.join(self.prep_folder, self.stack_filenames[i])
            stack_dict = {"image": path}
            stack_dict = loader(stack_dict)
            stack_dict = to_tensor(stack_dict)
            stack_dict = add_channel(stack_dict)
            #keep meta data correct
            stack_dict["image_meta_dict"]["spatial_shape"] = np.array(list(stack_dict["image"].shape)[1:])

            #move to gpu
            if to_device:
                stack_dict = to_device(stack_dict)
            stack_list.append(stack_dict)
        return stack_list
    
    
    def create_common_volume(self):
        """
        Combine updated local stacks that are in common coordinate system, to 
        one superpositioned volume

        Returns
        -------
        world_stack : dictionary
            contains nifti file of the the reconstructed brain.

        """
        stacks = self.load_stacks()
        to_device = monai.transforms.ToDeviced(keys = ["image"], device = self.device)
        tmp = t.zeros_like(stacks[0]["image"])
        
        
        for st in range(1,self.k):
            tmp = tmp + stacks[st]["image"]
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
            fixed_image = self.create_common_volume()
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
        return slices, n_slices


    
    
    def create_common_volume_registration(self):
        stacks = self.load_stacks()
        #to_device = monai.transforms.ToDeviced(keys = ["image"], device = self.device)
        
        stacks[0]["image"] = stacks[0]["image"].unsqueeze(0)
        fixed_meta = stacks[0]["image_meta_dict"]
        common_image = stacks[0]["image"].unsqueeze(0)
        
        for st in range(1,self.k):
            image = stacks[st]["image"].unsqueeze(0)
            meta = stacks[st]["image_meta_dict"]
            
            model = custom_models.Reconstruction(n_slices = 1, device = self.device)
            loss = loss_module.RegistrationLoss("ncc", self.device)
            optimizer = t.optim.Adam(model.parameters(), lr = 0.01)
            
            for ep in range(0,3):
                transformed = model(image.detach(), meta, fixed_meta)
                transformed = transformed.to(self.device)
                loss_tensor = loss(transformed, stacks[0])
                loss_tensor.backward()
                optimizer.step()
            transformed = transformed.detach()
            common_image = common_image + transformed
        
        return {"image":common_image.squeeze().unsqueeze(0), "image_meta_dict": fixed_meta}
    
    def optimize_multiple_stacks(self, epochs:int, inner_epochs:int, lr, loss_fnc = "ncc", opt_alg = "Adam"):
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
        for st in range (0,self.k):
            slice_tmp, n_slices = self.construct_slices_from_stack(self.ground_truth[st])
            slices.append(slice_tmp)
            #n_slices = self.ground_truth[st]["image"].shape[-1]
            models.append(custom_models.Reconstruction(n_slices = n_slices, device = self.device))
        
        #resampling_model = custom_models.ResamplingToFixed()
        #resampling_model.to(self.device)
        
        loss = loss_module.RegistrationLoss(loss_fnc, self.device)
    
        
        loss_log = np.zeros((epochs,self.k,inner_epochs))
        
        resampler = monai.transforms.ResampleToMatch(mode = self.mode)
        
        common_volume = t.zeros_like(self.fixed_images[0]["image"])
        
        for epoch in range(0,epochs):
            fixed_image = self.fixed_images[epoch]
            if epoch > 0:
                common_volume = common_volume.squeeze().unsqueeze(0)
                fixed_image["image"],_ = resampler(common_volume,src_meta=self.fixed_images[epoch - 1]["image_meta_dict"],
                                                 dst_meta = fixed_image["image_meta_dict"])
                
                fixed_image["image"] = fixed_image["image"].unsqueeze(0)
                common_volume = t.zeros_like(fixed_image["image"])
            
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
                    transformed_slices = model(slices_tmp.detach(), local_stack["image_meta_dict"], fixed_image["image_meta_dict"], transform_to_fixed = True, mode = self.mode)
                    transformed_slices = transformed_slices.to(self.device)
                    print(f'forward pass. {time.time() - timer} s ')
                    
                    dot = make_dot(transformed_slices[0,:,:,:,:], params = dict(model.named_parameters()))
                    
                    timer = time.time()
                    loss_tensor = loss(transformed_slices, fixed_image)
                    
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