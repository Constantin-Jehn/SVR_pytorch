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


class SVR_optimizer():
    def __init__(self,src_folder, prep_folder, stack_filenames, mask_filename, pixdim, device, mode):
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
            DESCRIPTION.
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
        self.pixdim = pixdim
        self.mode = mode
        
        self.crop_images(upsampling=True)
        self.fixed_image = self.create_common_volume()
        add_channel = AddChanneld(keys=["image"])
        
        to_device = monai.transforms.ToDeviced(keys = ["image"], device = self.device)
        self.fixed_image = add_channel(self.fixed_image)
        self.fixed_image["image"].requires_grad = False
        self.fixed_image = to_device(self.fixed_image)
        print("fixed_image_generated")
        
        
        self.initial_vol = {"image": deepcopy(self.fixed_image["image"]), "image_meta_dict": deepcopy(self.fixed_image["image_meta_dict"])}
        self.initial_vol = to_device(self.initial_vol)
        
        self.crop_images(upsampling = False)
        #remains in initial coordiate system
        self.ground_truth = self.load_stacks(to_device=True)
        print(f'preprocessing done in {time.time() - timer} s')
        #self.ground_truth, self.im_slices, self.target_dict, self.k = self.preprocess()


    def crop_images(self, upsampling = False):
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
                    #only upsample first stack, for remaining stack it'd done by resamplich to this stack
                    upsampler = tio.transforms.Resample(self.pixdim)
                    cropped_stack = upsampler(cropped_stack)
                else:
                    path_stack = os.path.join(self.prep_folder, self.stack_filenames[0])
                    resampler = tio.transforms.Resample(path_stack)
                    cropped_stack = resampler(cropped_stack)

            path_dst = os.path.join(self.prep_folder, filename)
            cropped_stack.stack.save(path_dst)
            
        
        #create a common place for the reconstruction
        common_image = tio.Image(tensor = t.zeros((1,100,100,100)))
        path_dst = os.path.join(self.prep_folder, "world.nii.gz")
        common_image.save(path_dst)
    
    def resample_to_pixdim(self):
        """
        After cropping and having saved the ground truth image, bring the images
        to desired resolution for reconstruction

        Returns
        -------
        None.
        """
        resampler = tio.transforms.Resample(self.pixdim)
        
        for i in range(0,self.k):
            filename = self.stack_filenames[i]
            path_stack = os.path.join(self.prep_folder, filename)
            stack = tio.ScalarImage(path_stack)
            resampled_stack = resampler(stack)
            path_dst = os.path.join(self.prep_folder, filename)
            resampled_stack.save(path_dst)    
    
    
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
        
        n_slices = stack_image.shape[-1]
        
        slices = t.zeros_like(stack_image).repeat(n_slices,1,1,1,1)
        for i in range (0,n_slices):
            tmp = deepcopy(stack_image)
            tmp[:,:,:,:,:i] = 0
            tmp[:,:,:,:,i+1:] = 0
            slices[i,:,:,:,:] = tmp
        return slices

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
        for st in range (0,self.k):
            n_slices = self.ground_truth[st]["image"].shape[-1]
            models.append(custom_models.Reconstruction(n_slices = n_slices, device = self.device))
        
        #resampling_model = custom_models.ResamplingToFixed()
        #resampling_model.to(self.device)
        
        loss = loss_module.RegistrationLoss(loss_fnc, self.device)
        
        loss_log = np.zeros((epochs,self.k,inner_epochs))
        
        
        
        for epoch in range(0,epochs):
            print(f'\n\n Epoch: {epoch}')
            
            for st in range(0,self.k):
                #loss_stack = list()
                print(f"\n  stack: {st}")
                model = models[st]
                model.to(self.device)
                
                #in batch first shape
                slices_tmp = self.construct_slices_from_stack(self.ground_truth[st])
                
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
                    transformed_slices = model(slices_tmp.detach(), local_stack["image_meta_dict"], self.fixed_image["image_meta_dict"])
                    transformed_slices = transformed_slices.to(self.device)
                    print(f'forward pass. {time.time() - timer} s ')
                    
                    timer = time.time()
                    loss_tensor = loss(transformed_slices, self.fixed_image)
                    print(f'loss:  {time.time() - timer} s ')
                    #loss_stack.append(loss_tensor.item())
                    
                    #here stack[st] is in coordinates of fixed image
                    #print(f'Epoch: {epoch} loss: {loss_tensor.item()}')
                    timer = time.time()
                    if inner_epoch == (inner_epochs - 1) :
                        rt_graph = False
                    else:
                        rt_graph = True
                    
                    loss_tensor.backward(retain_graph = False)
                    print(f'backward:  {time.time() - timer} s ')
                    print(f'loss: {loss_tensor.item()}')
                    loss_log[epoch,st,inner_epoch] = loss_tensor
                    timer = time.time()
                    optimizer.step()
                    print(f'optimizer:  {time.time() - timer} s ')

            transformed_slices = transformed_slices.detach()
            
            #update common_volume
            self.fixed_image["image"] = t.sum(transformed_slices, dim = 0).unsqueeze(0)
            
            #for name, param in model.named_parameters():
               # if param.requires_grad:
                    #print (f'parameter {name} has value not equal to zero: {t.all(param.data == 0)}')
            
        #cast to image format
        self.fixed_image["image"] = t.squeeze(self.fixed_image["image"]).unsqueeze(0)
        
        loss_log = 0 
        return self.fixed_image, loss_log


    def bending_loss_fucntion_single_stack(target_dict_image):
        monai_bending = monai.losses.BendingEnergyLoss()
        return monai_bending(target_dict_image.expand(-1,3,-1,-1,-1))