import torchio as tio
import monai
from monai.transforms import (
    AddChanneld,
    LoadImaged,
    ToTensord,
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
from SVR_outlier_removal import Outlier_Removal_Slices_cste, Outlier_Removal_Voxels, Outlier_Removal_Slices
from torch.utils.tensorboard import SummaryWriter

import SimpleITK as sitk

from SVR_Evaluation import psnr

class SVR_optimizer():
    def __init__(self, src_folder:str, prep_folder:str, result_folder:str, stack_filenames:list, mask_filename:str, pixdims:list, device:str, PSF, monai_mode:str, tio_mode:str, roi_only:bool=False)->None:
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
            sav_gol_kernel_size(int): kernel support for Savitzky Golay Filter
            sav_gol_order(int): Polynomial order for interpolation of Savitzky Golay Filter
            roi_only(bool,optional): whether to return only the region of interest, and set remaining voxels to zero
        """
        timer = time.time()
        
        self.device = device
        print(f'Program runs on: {self.device}')
        self.stack_filenames = stack_filenames
        self.k = len(self.stack_filenames)
        self.mode = monai_mode
        self.pixdims = pixdims
        
        self.svr_preprocessor = Preprocesser(src_folder, prep_folder, result_folder, stack_filenames, mask_filename, device, monai_mode, tio_mode)
        
        self.fixed_image, self.stacks, self.slice_dimensions = self.svr_preprocessor.preprocess_stacks_and_common_vol(self.pixdims[0], PSF,roi_only=roi_only)
        
        self.ground_truth = self.stacks

        self.tio_mode = tio_mode

        self.result_folder = result_folder



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



    def optimize_volume_to_slice(self, epochs:int, inner_epochs:int, lr, PSF, lambda_scheduler, loss_fnc = "ncc", opt_alg = "Adam", tensorboard:bool = False, tensorboard_path = '', from_checkpoint:bool=False, last_rec_file:str='', last_epoch:int=0):
        """
        optimizes transform of individual slices to mitigate motion artefact, uses initial 3d-3d registration
        implemented in SVR_Preprocessor

        Args:
            epochs (int): epochs of registration of all stacks
            inner_epochs (int): epochs of 3d-2d registration of each stack
            lr (_type_): learning rate of optimizer
            PSF(functio): Point spread function
            lambda_scheduler(lambda-expression): controls learning rate schedule
            loss_fnc (str, optional): loss function Defaults to "ncc".
            opt_alg (str, optional): optimization algorithm Defaults to "Adam"
            tensorboard(bool):whether or not to write to tensorboard
            from_checkpoint(bool):whether to start from checkpoint
            last_rec_file(string): if starting from checkpoint name of last reconstruction file to start from
        """
        writer = SummaryWriter(tensorboard_path)
        #Afffine transformations for updating common volume from slices (use bilinear because it's 2d transform)
        affine_transform_slices = monai.networks.layers.AffineTransform(mode = "bilinear",  normalized = True, padding_mode = "zeros")

            
        models, optimizers, losses, schedulers, affines_slices, n_slices, slices, slice_dims = self.prepare_optimization(PSF, lambda_scheduler, opt_alg, loss_fnc, lr)
        #loss = loss_module.Loss_Volume_to_Slice(loss_fnc, self.device)
        first_epoch = 0
        if from_checkpoint:
            models, optimizers = self.load_models_and_optimizers(PSF, lr)
            ###load fixed_image_tensor
            add_channel = AddChanneld(keys=["image"])
            loader = LoadImaged(keys=["image"])
            to_tensor = ToTensord(keys=["image"])

            path = os.path.join(self.result_folder, last_rec_file)
            fixed_image = {"image": path}
            fixed_image = loader(fixed_image)
            fixed_image = to_tensor(fixed_image)
            fixed_image = add_channel(fixed_image)
            fixed_image = add_channel(fixed_image)

            self.fixed_image = fixed_image

            first_epoch = last_epoch + 1

        fixed_image_tensor = self.fixed_image["image"]
        fixed_image_meta = self.fixed_image["image_meta_dict"]
        
        #use this template for tio-resampling operations of stacks during update
        tio_fixed_image_template = self.svr_preprocessor.monai_to_torchio(self.fixed_image)
        resampling_to_fixed_tio = tio.transforms.Resample(tio_fixed_image_template, image_interpolation=self.tio_mode)

        for epoch in range(first_epoch,first_epoch + epochs):
            common_volume = t.zeros_like(self.fixed_image["image"], device=self.device)
            #used to compare to absence of outlier removal
            common_volume_pure = t.zeros_like(self.fixed_image["image"], device=self.device)
            tio_fixed_image_template = self.svr_preprocessor.monai_to_torchio({"image": fixed_image_tensor, "image_meta_dict": fixed_image_meta})
            resampling_to_fixed_tio = tio.transforms.Resample(tio_fixed_image_template, image_interpolation=self.tio_mode)
            print(f'\n\n Epoch: {epoch}')
            
            for st in range (0, self.k):
                print(f"\n  stack: {st}")
                #each stack has its own model + optimizer
                model = models[st]
                optimizer = optimizers[st]
                scheduler = schedulers[st]
                loss = losses[st]

                local_stack = self.stacks[st]
                local_slices = slices[st]

                local_stack_tio = self.svr_preprocessor.monai_to_torchio(local_stack)

                fixed_image_resampled_tensor = self.svr_preprocessor.resample_fixed_image_to_local_stack(fixed_image_tensor, fixed_image_meta["affine"], local_stack_tio.tensor,
                                                                                            local_stack_tio.affine)
                
                if st == 0:
                    print(f'learning rate: {optimizer.param_groups[0]["lr"]}')

                #optimization procedure
                for inner_epoch in range(0,inner_epochs):
                    model.train()
                    optimizer.zero_grad()

                    #return fixed_images resamples to local stack where inverse affines were applied
                    #in shape (n_slices,1,[stack_shape]) affines 
                    tr_fixed_images, affines_tmp = model(fixed_image_resampled_tensor.detach())

                    tr_fixed_images = tr_fixed_images.to(self.device)

                    loss_tensor = loss(tr_fixed_images, local_slices, n_slices[st], slice_dims[st])
                    print(f'loss: {loss_tensor.item()}')

                    loss_tensor.backward(retain_graph = False)

                    optimizer.step()

                    #visualization of loss in tensorboard
                    if tensorboard:
                        """
                        if epoch == 0 and st == 0:
                            model_tensor_board = custom_models.Volume_to_Slice(n_slices=2, device=self.device, mode = self.mode, tio_mode = self.tio_mode)
                            red_input = fixed_image_tensor[:,:,0:2,:,:].detach()
                            writer.add_graph(model_tensor_board,(red_input, t.tensor(fixed_image_meta["affine"]), local_stack_tio.tensor, t.tensor(local_stack_tio.affine)))
                            writer.close
                        """
                    #calcuates 2d between a local slice and the corresponding slice in the tr_fixed_image
                        if st == 0:
                            writer.add_scalar(f"Learning_rate", optimizer.param_groups[0]["lr"], epoch)

                        if inner_epoch == 0:
                            writer.add_scalar(f"Loss_stack_{st}", loss_tensor.item(), epoch)


                    """
                    torchviz.make_dot(loss_tensor, params= dict(model.named_parameters()))
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            print (f'parameter {name} has only zero values: {t.all(param.data == 0)}')
                    """
                

                #update procedure
                timer = time.time()

                for sl in range(0,n_slices[st]):
                    affines_slices[st][sl,:,:] = affines_tmp[sl]
                
                timer = time.time()
                
                #uses difference to now transformed fixed images
                error_tensor = self.get_error_tensor(tr_fixed_images, local_slices, n_slices[st], slice_dims[st])
                
                #error_tensor = (fixed_image_resampled_tensor - local_stack_tio.tensor).repeat(n_slices[st],1,1,1,1)
                #outlier removal
                likelihood_images_voxels = self.outlier_removal_voxels(error_tensor, n_slices[st], slice_dims[st])
                likelihood_images_slices = self.outlier_removal_slices(likelihood_images_voxels, n_slices[st], slice_dims[st])
                local_slices_outlier_removed = t.mul(local_slices, t.mul(likelihood_images_voxels, likelihood_images_slices))
                #local_slices = t.mul(local_slices, likelihood_images_voxels)
                print(f'outlier removal:  {time.time() - timer} s ')

                timer = time.time()
                #multiply likelihood to each voxel for outlier removal
                
                affines_tmp = affines_slices[st]
                #apply affines to transform slices
                transformed_slices = affine_transform_slices(local_slices_outlier_removed, affines_tmp)
                transformed_slices = transformed_slices.detach()
                
                #update current stack from slices
                common_volume = self.update_common_volume_from_slices(common_volume,transformed_slices, n_slices, st, local_stack["image_meta_dict"]["affine"], PSF, resampling_to_fixed_tio, fixed_image_meta)
                #to compare outlier unremoved volume
                """
                common_volume_pure = self.update_common_volume_from_slices(common_volume,affine_transform_slices(local_slices, affines_tmp).detach(), n_slices, st, local_stack["image_meta_dict"]["affine"], PSF, resampling_to_fixed_tio, fixed_image_meta)
                """
            
            if t.std(common_volume) != 0:
                normalizer = tv.transforms.Normalize(t.mean(common_volume), t.std(common_volume))
                common_volume = normalizer(common_volume)
            
            #to compare outlier unremoved volume
            """
            if t.std(common_volume_pure) != 0:
                normalizer = tv.transforms.Normalize(t.mean(common_volume_pure), t.std(common_volume_pure))
                common_volume_pure = normalizer(common_volume_pure)
            """

            #update fixed_image
            timer = time.time()
            fixed_image_tensor = common_volume

            if epoch < epochs - 1:
                #not last epoch yet --> upsample fixed_image if wanted
                upsample_bool = (self.pixdims[epoch + 1] == self.pixdims[epoch])
                fixed_image = self.svr_preprocessor.save_intermediate_reconstruction_and_upsample(fixed_image_tensor, fixed_image_meta, epoch, upsample=upsample_bool, pix_dim = self.pixdims[epoch+1])

                #benchmark without outlier removal
                """
                _ = self.svr_preprocessor.save_intermediate_reconstruction_and_upsample(common_volume_pure, fixed_image_meta, int('2' + str(epoch)), upsample=upsample_bool, pix_dim = self.pixdims[epoch+1])
                """
                fixed_image_tensor = fixed_image["image"]
                fixed_image_meta = fixed_image["image_meta_dict"]
                common_volume = t.zeros_like(fixed_image_tensor)
            else:
                self.svr_preprocessor.save_intermediate_reconstruction(fixed_image_tensor,fixed_image_meta,epoch)
            print(f'fixed_volume update:  {time.time() - timer} s ')

            fixed_dict = {"image": fixed_image_tensor, "image_meta_dict": fixed_image_meta}
            print(f'PSNR: {psnr(fixed_dict,self.stacks,n_slices, self.tio_mode)}')

            for st in range(0, self.k):
                scheduler = schedulers[st]
                scheduler.step()
            
            self.save_models_and_optimizers(models, optimizers)
            
        writer.close()

    def get_error_tensor(self,tr_fixed_images:t.tensor, local_slices:t.tensor, n_slices:int, slice_dim:int)->t.tensor:
        """_summary_

        Args:
            tr_fixed_images (t.tensor): updated and transformed fixed image
            local_slices (t.tensor): initial slices of current stack
            n_slices (int): number of slices
            slice_dim (int): dimension along which slices are taken

        Returns:
            t.tensor: error_tensor
        """
        error_tensor = t.zeros_like(tr_fixed_images)
        for sl in range (0,n_slices):
            #outlier removal
            if slice_dim == 0:
                pred = tr_fixed_images[sl,0,sl,:,:]
                target = local_slices[sl,0,sl,:,:]
                error_tensor[sl,0,sl,:,:] = pred - target
            elif slice_dim == 1:
                pred = tr_fixed_images[sl,0,:,sl,:]
                target = local_slices[sl,0,:,sl,:]
                error_tensor[sl,0,:,sl,:] = pred - target
            elif slice_dim == 2:
                pred = tr_fixed_images[sl,0,:,:,sl]
                target = local_slices[sl,0,:,:,sl]
                error_tensor[sl,0,:,:,sl] = pred - target

        return error_tensor

    def outlier_removal_voxels(self, error_tensor:t.tensor,  n_slices:int, slice_dim:int) -> t.tensor:
        """_summary_

        Args:
            error_tensor (t.tensor): Error tensor between transformed fixed image and local slices
            n_slices (int): number of slices
            slice_dim (int): dimension along which slices are taken

        Returns:
            t.tensor: likelihood image of voxel being inlier
        """
        likelihood_images = t.zeros_like(error_tensor, device=self.device)
        for sl in range(0,n_slices):
            outlier_remover = Outlier_Removal_Voxels()

            if slice_dim == 0:
                p_voxels = outlier_remover(error_tensor[sl,0,sl,:,:])
                likelihood_images[sl,0,sl,:,:] = p_voxels
            elif slice_dim == 1:
                p_voxels = outlier_remover(error_tensor[sl,0,:,sl,:])
                likelihood_images[sl,0,:,sl,:] = p_voxels
            elif slice_dim == 2:
                p_voxels = outlier_remover(error_tensor[sl,0,:,:,sl])
                likelihood_images[sl,0,:,:,sl] = p_voxels
        return likelihood_images

    def outlier_removal_slices(self, likelihood_images:t.tensor, n_slices:int, slice_dim:int)->t.tensor:

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
        
        outlier_remover = Outlier_Removal_Slices_cste(red_voxel_prob)
        
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

    def update_common_volume_from_slices(self, common_volume:t.tensor, transformed_slices:t.tensor, n_slices:list, st:int, local_stack_affine:t.tensor, PSF, resampler, fixed_image_meta:dict)->t.tensor:
        #update current stack from slices
                common_stack = t.zeros_like(common_volume, device=self.device)
                for sl in range(0,n_slices[st]):
                    slice_tmp = transformed_slices[sl,:,:,:,:]
                    slice_tmp_tio = tio.Image(tensor=slice_tmp.squeeze().unsqueeze(0).detach().cpu(), affine=local_stack_affine)
                    slice_tio_transformed = resampler(slice_tmp_tio)

                    #Gaussian kernel over each slice
                    #tio_transformed_blurred = self.gaussian_smoother(slice_tio_transformed.tensor)
                    #Use golay filter as PSF
                    tio_transformed_blurred = PSF(slice_tio_transformed.tensor)

                    common_stack = common_stack + tio_transformed_blurred.unsqueeze(0).to(self.device)
                    #common_stack = common_stack + tio_transformed.tensor.unsqueeze(0).to(self.device)    

                #PSNR output
                fixed_dict = {"image": common_stack, "image_meta_dict": fixed_image_meta}
                print(f'PSNR: {psnr(fixed_dict,self.stacks,n_slices, self.tio_mode)}')

                #update common volume from stack
                common_volume = common_volume + common_stack
                return common_volume

    def prepare_optimization(self, PSF,lambda1, opt_alg, loss_fnc, lr):
        """
        Prepar optimization, generate slices, load models, optimizers and scheduler

        Args:
            PSF (_type_): 
            lambda1 (lambda-expression):
            opt_alg (str): "Adam" or "SGD"
            loss_fnc (str): "ncc" or "mi"
            lr (float): learning rate

        Returns:
            tuple: models, optimizers, losses, schedulers, affines_slices, n_slices, slices, slice_dims
        """
        slices = list()
        n_slices = list()
        slice_dims = self.slice_dimensions
        affines_slices = list()
    
        models = list()
        optimizers = list()
        schedulers = list()
        losses = list()

        for st in range(0,self.k):
            slice_tmp, n_slice = self.construct_slices_from_stack(self.stacks[st], slice_dims[st])
            slices.append(slice_tmp)
            n_slices.append(n_slice)
            model_stack = custom_models.Volume_to_Slice(PSF, n_slices=n_slices[st], device=self.device, mode = self.mode, tio_mode = self.tio_mode)
            model_stack.to(self.device)
            models.append(model_stack)

            #set kernel size to smaller shape of stack
            kernel_size = min(self.stacks[st]["image"].shape[1], self.stacks[st]["image"].shape[2])
            #kernel_size = 31
            loss = loss_module.Loss_Volume_to_Slice(kernel_size, loss_fnc, self.device)
            losses.append(loss)

            if opt_alg == "SGD":
                optimizer = t.optim.SGD(model_stack.parameters(), lr = lr)
            elif(opt_alg == "Adam"):
                optimizer = t.optim.Adam(model_stack.parameters(), lr = lr)
            else:
                assert("Choose SGD or Adam as optimizer")
            
            optimizers.append(optimizer)

            scheduler = t.optim.lr_scheduler.LambdaLR(optimizer,lambda1)
            schedulers.append(scheduler)

            #store affine transforms
            affines_slices.append(t.eye(4, device=self.device).unsqueeze(0).repeat(n_slice,1,1))

        return models, optimizers, losses, schedulers, affines_slices, n_slices, slices, slice_dims

    def save_models_and_optimizers(self, models:list, optimizers:list):
        """Saves models and optimizers as checkpoint
        referenc: https://pytorch.org/tutorials/beginner/saving_loading_models.html#warmstarting-model-using-parameters-from-a-different-model

        Args:
            models (list): list of the models
            optimizers (list): list of the optimizers
        """
        model_dict = {}
        n_models = len(models)
        model_dict["n_models"] = n_models

        for i in range(0,n_models):
            model_str = "model_" + str(i) +"_state_dict"
            opt_str = "optimizer_" + str(i) + "_state_dict"
            n_slices_str = "n_slices_" + str(i)
            lr_str = "lr_" + str(i)
            model_dict[n_slices_str] = models[i].n_slices
            model_dict[model_str] = models[i].state_dict()
            model_dict[lr_str] = optimizers[i].param_groups[0]["lr"]
            model_dict[opt_str] = optimizers[i].state_dict()
        PATH = os.path.join(self.result_folder,"models_optimizers.pt")

        t.save(model_dict,PATH)
        

    def load_models_and_optimizers(self, PSF, lr)->tuple:

        PATH = os.path.join(self.result_folder,"models_optimizers.pt")
        checkpoint = t.load(PATH, map_location=self.device)
        n_models = checkpoint["n_models"]
        models, optimizers = list(), list()
        
        for i in range(0,n_models):
            n_slices_str = "n_slices_" + str(i)
            model_str = "model_" + str(i) +"_state_dict"
            opt_str = "optimizer_" + str(i) + "_state_dict"
            lr_str = "lr_" + str(i)

            n_slices = checkpoint[n_slices_str]
            model = custom_models.Volume_to_Slice(PSF, n_slices=n_slices, device=self.device, mode = self.mode, tio_mode = self.tio_mode)
            model.load_state_dict(checkpoint[model_str])
            model.train()
            
            #lr = checkpoint[lr_str]
            optimizer = t.optim.Adam(model.parameters(), lr = lr)
            optimizer.load_state_dict(checkpoint[opt_str])

            models.append(model)
            optimizers.append(optimizer)
        
        return models, optimizers



"""
               for name, param in model.named_parameters():
                if param.requires_grad:
                    print (f'parameter {name} has value not equal to zero: {t.all(param.data == 0)}')


            timer = time.time()
                    optimizer.step()
                    print(f'optimizer:  {time.time() - timer} s ')
"""

"""
if epoch == 0 and st == 0:
    model_tensor_board = loss_module.Loss_Volume_to_Slice(loss_fnc, self.device)
    red_fixed, red_local = tr_fixed_images[0:2,:,:,:,0:2].detach(), local_slices[0:2,:,:,:,0:2]
    self.writer.add_graph(model_tensor_board,(red_fixed, red_local,t.tensor(2.0),t.tensor(2.0)))
    self.writer.close
"""