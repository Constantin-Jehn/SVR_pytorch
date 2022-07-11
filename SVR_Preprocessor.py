
from fileinput import filename
import torchio as tio
import monai

from monai.transforms import (
    AddChanneld,
    LoadImaged,
    ToTensord,
)
import os
import numpy as np
#from zmq import device
import custom_models
import torch as t
from copy import deepcopy
import loss_module
import time
import SimpleITK as sitk
import torchvision as tv
from SVR_outlier_removal import Outlier_Removal_Slices_cste, Outlier_Removal_Voxels
import utils
#from torch.utils.tensorboard import SummaryWriter

class Preprocesser():
    def __init__(self, src_folder:str, prep_folder:str, result_folder:str, stack_filenames:list, mask_filename:str, device:str, monai_mode:str, tio_mode:str)->None:
        """
        Constructor of Preprocessor: class to take care of preprocessing such as initial 3d-3d registration, denoising or normalization

        Args:
            src_folder (str): initial nifti_files
            prep_folder (str): folder to save prepocessed files
            result_folder (str): folder to save reconstruction results
            stack_filenames (list): of filenames of stacks to be reconstructed
            mask_filename (str): nifti filename to crop input images
            device (str): device to calculate on mainly
            monai_mode (str): interpolation mode for monai resampling
            tio_mode (str): interpolation mode for monai resampling
        """
        self.device = device
        self.src_folder = src_folder
        self.prep_folder = prep_folder
        self.result_folder = result_folder
        
        self.k = len(stack_filenames)
        
        self.mask_filename = mask_filename
        self.mode = monai_mode
        self.tio_mode = tio_mode

        #orders stack filenames according to motion corruption
        #selects four stacks with least corruptions
        self.stack_filenames = self.order_stackfilenames_for_preregistration(stack_filenames)
        self.k = len(self.stack_filenames)
        #self.writer = SummaryWriter("runs/test_session")

    def preprocess_stacks_and_common_vol(self, init_pix_dim:tuple, PSF, save_intermediates:bool=False, roi_only:bool = False, lr_vol_vol:float = 0.0035)->tuple:
        """        
        preprocessing procedure before the optimization contains:
        denoising, normalization, initial 3d-3d registration

        Args:
            init_pix_dim (tuple): initial resolution of the fixed image
            PSF(function): point spread function
            loss_kernel_size(int/str): kernel size of local norm. cc
            save_intermediates (bool, optional):whether to save intermediate steps of preprocessing. Default to False.
            roi_only(bool,optional): whether to return only the region of interest, and set remaining voxels to zero
            lr_vol_vol(float): learning rate for volume-volume pregistration

        Returns:
            tuple: 
                initial fixed volume(monai dict), 
                pre processed stacks(list of monai dicts),
                slice_dimensions (list)
                rot_params (list): inital rotation parameters of pre registration
                trans_params(list): initial translation parameters of pre registration
        """

        #to_device = monai.transforms.ToDeviced(keys = ["image"], device = self.device)
        slice_dimensions = self.crop_images(upsampling=False,roi_only=roi_only)

        # load cropped stacks
        stacks = self.load_stacks(to_device=True)
        # denoise stacks
        stacks = self.denoising(stacks)

        if save_intermediates:
            stacks = utils.save_stacks(stacks, 'den', self.mode)

        stacks = self.bias_correction_sitk(stacks)

        if save_intermediates:
            stacks = utils.save_stacks(stacks, 'bias', self.mode)

        stacks = self.normalize(stacks)
        #stacks = self.histogram_normalize_stacks(stacks)
        if save_intermediates:
            stacks = utils.save_stacks(stacks, 'norm', self.mode)

        stacks_preprocessed = utils.resample_stacks(stacks, init_pix_dim, self.tio_mode)

        fixed_image, stacks, rot_params, trans_params = self.create_common_volume_registration(stacks_preprocessed, PSF, lr_vol_vol)

        #stacks = self.outlier_removal(fixed_image, stacks)

        
        fixed_image = utils.resample_fixed_image(fixed_image, init_pix_dim,self.result_folder,self.mode,self.tio_mode)

        if save_intermediates:
            stacks = utils.save_stacks(stacks, 'reg', self.mode)

        for st in range(0, len(stacks)):
            stacks[st]["image"] = stacks[st]["image"].squeeze().unsqueeze(0)

        return fixed_image, stacks, slice_dimensions, rot_params, trans_params

    def get_cropped_stacks(self)->list:
        """
        crops images and returns them

        Returns:
            list: list of cropped stacks
        """
        self.crop_images(upsampling=False)
        # load cropped stacks
        stacks = self.load_stacks(to_device=True)
        return stacks

    def crop_images(self, upsampling:bool=True, pixdim=1.0, roi_only = False)->None:
        """
        crops images from source directory according to provided mask and saves them to prep folder

        Args:
            upsampling (bool, optional): whether or not to upsample fixed image Defaults to False.
            pixdim (int, optional): pix dim to upsample to. Defaults to 0.
        Returns:
            slice_dimensions (list): For each stack the index of the dimension in which to slice
        """
        path_mask = os.path.join(self.src_folder, self.mask_filename)
        mask = tio.LabelMap(path_mask)
        path_dst = os.path.join(self.prep_folder, self.mask_filename)
        mask.save(path_dst)

        slice_dimensions = list()

        for i in range(0, self.k):
            #resample mask to each stack
            filename = self.stack_filenames[i]
            path_stack = os.path.join(self.src_folder, filename)
            stack = tio.ScalarImage(path_stack)

            if upsampling:
                upsampler = tio.transforms.Resample(pixdim, image_interpolation = self.tio_mode)
                stack = upsampler(stack)

            #book keeping of dimensionality of slice dimensions e.g. later for def of transformation model
            slice_dimensions.append(list(stack.tensor.shape[1:]).index(min(list(stack.tensor.shape[1:]))))

            resampler = tio.transforms.Resample(stack)
            resampled_mask = resampler(deepcopy(mask))

            #sets non masked values to zero
            if roi_only:
                stack_tensor = stack.tensor * resampled_mask.tensor
                stack.set_data(stack_tensor)
              
            subject = tio.Subject(stack=stack, mask=resampled_mask)

            #find indices to crop image as tensor
            masked_indices_1 = t.nonzero(subject.mask.data)
            min_indices = np.array([t.min(masked_indices_1[:, 1]).item(), t.min(
                masked_indices_1[:, 2]).item(), t.min(masked_indices_1[:, 3]).item()])
            max_indices = np.array([t.max(masked_indices_1[:, 1]).item(), t.max(
                masked_indices_1[:, 2]).item(), t.max(masked_indices_1[:, 3]).item()])
            roi_size = (max_indices - min_indices)

            cropper = tio.CropOrPad(list(roi_size), mask_name='mask')

            cropped_stack = cropper(subject)

            """
            if upsampling:
                if i == 0:
                    # only upsample first stack, for remaining stack it's done by resamplich to this stack
                    upsampler = tio.transforms.Resample(pixdim, image_interpolation = self.tio_mode)
                    cropped_stack = upsampler(cropped_stack)
                else:
                    path_stack = os.path.join(
                        self.prep_folder, self.stack_filenames[0])
                    resampler = tio.transforms.Resample(path_stack, image_interpolation = self.tio_mode)
                    cropped_stack = resampler(cropped_stack)
            """

            path_dst = os.path.join(self.prep_folder, filename)

            #to see if it'll be in standard planes
            #cropped_stack.stack.affine(np.eye(4))
            cropped_stack.stack.save(path_dst)

        return slice_dimensions

    def load_stacks(self, to_device=False)->list:
        """
        After cropping the initial images in low resolution are saved in their original coordinates
        for the loss computation

        Args:
            to_device (bool, optional): whether o put tensor to device Defaults to False.

        Returns:
            list: _list of dictionaries containing the ground truths
        """

        add_channel = AddChanneld(keys=["image"])
        loader = LoadImaged(keys=["image"])
        to_tensor = ToTensord(keys=["image"])
        if to_device:
            to_device = monai.transforms.ToDeviced(
                keys=["image"], device=self.device)

        stack_list = list()
        for i in range(0, self.k):
            path = os.path.join(self.prep_folder, self.stack_filenames[i])
            stack_dict = {"image": path}
            stack_dict = loader(stack_dict)
            stack_dict = to_tensor(stack_dict)
            stack_dict = add_channel(stack_dict)
            # keep meta data correct
            stack_dict["image_meta_dict"]["spatial_shape"] = np.array(
                list(stack_dict["image"].shape)[1:])
            stack_dict["image_meta_dict"]["filename_or_obj"] = self.stack_filenames[i]
            # move to gpu
            if to_device:
                stack_dict = to_device(stack_dict)
            stack_list.append(stack_dict)
        return stack_list


    def denoising(self, stacks:list)->list:
        """Applies Gaussian Sharpen Filter to all stacks

        Args:
            stacks (list): unprocessed stacks

        Returns:
            list: denoised stacks
        """

        gauss_sharpen = monai.transforms.GaussianSharpen(
            sigma1=1, sigma2=1, alpha=3)
        for st in range(0, self.k):
            stacks[st]["image"] = gauss_sharpen(stacks[st]["image"])
        return stacks

    def denoise_single_tensor(self, single_slice_image:t.tensor)->t.tensor:
        """
        denoises single tensor

        Args:
            single_slice_image (t.tensor): 

        Returns:
            t.tensor:
        """
        gauss_sharpen = monai.transforms.GaussianSharpen(
            sigma1=1, sigma2=1, alpha=3)
        return gauss_sharpen(single_slice_image)

    def bias_correction_sitk(self, stacks:list)->None:
        """Applies N4 Bias Correction and saves corrected files

        Args:
            stacks (list): stacks to be bias corrected
        """

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        for st in range(0, self.k):
            path = os.path.join(self.prep_folder,stacks[st]["image_meta_dict"]["filename_or_obj"])
            tio_image = utils.monai_to_torchio(stacks[st])
            tio_image.save(path)
            image = sitk.ReadImage(path, sitk.sitkFloat32)
            denoised_image = corrector.Execute(image)
            sitk.WriteImage(denoised_image, path)
        
        stacks = self.load_stacks(to_device=True)

        return stacks


    def create_common_volume_registration(self, stacks:list, PSF, lr_vol_vol)->tuple:
        """
        creates common volume and return registered stacks
        Args:
            stacks (list): initial stacks
            PSF(functio): point spread function
            
        Returns:
            tuple: inital fixed image, list of preregistered stacks, list of Rotation parameters of preregistration, list of translation parameters of preregistration
        """

        folder = "preprocessing"
        path = os.path.join(folder)
        #to_device = monai.transforms.ToDeviced(keys = ["image"], device = self.device)
        stacks[0]["image"] = stacks[0]["image"].unsqueeze(0)
        fixed_meta = deepcopy(stacks[0]["image_meta_dict"])
        fixed_meta["filename_or_obj"] = "reconstruction_volume.nii.gz"
        common_tensor = stacks[0]["image"].unsqueeze(0)

        affine_transform_monai = monai.networks.layers.AffineTransform(mode = "bilinear",  normalized = True, padding_mode = "zeros")


        tio_common_image = utils.monai_to_torchio(stacks[0])
        resample_to_common = tio.transforms.Resample(tio_common_image, image_interpolation=self.tio_mode)

        rot_params, trans_params = list(), list()
        #adds zeros for inital stack (used as template)
        rot_params.append(t.zeros(3,device=self.device))
        trans_params.append(t.zeros(3,device=self.device))

        #stacks[0]["image"] = utils.normalize_zero_to_one(stacks[0]["image"])

        for st in range(1, self.k):
            stack_tensor = stacks[st]["image"]
            stack_meta = stacks[st]["image_meta_dict"]

            model = custom_models.Volume_to_Volume(PSF, device=self.device)
            loss = loss_module.Loss_Volume_to_Volume("ncc", self.device)
            optimizer = t.optim.Adam(model.parameters(), lr=lr_vol_vol)

            fixed_image_resampled_tensor = utils.resample_fixed_image_to_local_stack(common_tensor,fixed_meta["affine"],stack_tensor,stack_meta["affine"],self.tio_mode,self.device)
            
            stack_tensor = stacks[st]["image"].unsqueeze(0)
            for ep in range(0, 18):
                transformed_fixed_tensor, affine_tmp = model(fixed_image_resampled_tensor)
                transformed_fixed_tensor = transformed_fixed_tensor.to(self.device)
                loss_tensor = loss(transformed_fixed_tensor,stack_tensor)
                loss_tensor.backward()

                #self.writer.add_scalar(f"preregistrations_{st}", loss_tensor.item(), ep)

                if ep < 14:
                    optimizer.step()

            transformed_fixed_tensor = transformed_fixed_tensor.detach()

            #remove outlier
            #stacks[st] = self.outlier_removal(transformed_fixed_tensor,stacks[st])
            #stack_tensor = stacks[st]["image"].unsqueeze(0)

            #comment out for control
            stacks[st]["image"] = affine_transform_monai(stack_tensor, affine_tmp)

            tio_stack = utils.monai_to_torchio(stacks[st])
            
            tio_stack = resample_to_common(tio_stack)

            stacks[st] = utils.update_monai_from_tio(tio_stack,stacks[st],stacks[st]["image_meta_dict"]["filename_or_obj"], self.device)

            rot_params_tmp, trans_params_tmp = model.get_parameters()

            rot_params.append(rot_params_tmp)
            trans_params.append(trans_params_tmp)

            common_tensor = common_tensor + stacks[st]["image"]

            #stacks[st]["image"] = utils.normalize_zero_to_one(stacks[st]["image"])

        common_tensor = utils.normalize_zero_to_one(common_tensor)
        #normalizer = tv.transforms.Normalize(t.mean(common_tensor), t.std(common_tensor))
        #common_tensor = normalizer(common_tensor)
        #common_tensor = t.div(common_tensor, t.max(common_tensor)/2047)

        return {"image": common_tensor.squeeze().unsqueeze(0).unsqueeze(0), "image_meta_dict": fixed_meta}, stacks, rot_params, trans_params

    def outlier_removal(self, transformed_fixed_tensor:t.tensor, stack:dict):
        """
        outlier removal during initial registration

        Args:
            transformed_fixed_image (dict): common volume after 3d-3d registration
            stacks (list): list of stacks to be registered
        """
        transformed_fixed_tensor = transformed_fixed_tensor.squeeze().unsqueeze(0)
        stack_tensor =  stack["image"]
        likelihood_image = t.zeros_like(stack_tensor)
        n_slices = stack_tensor.shape[-1]

        outlier_remover = Outlier_Removal_Voxels()

        for sl in range(0,n_slices):
            error_tensor = transformed_fixed_tensor[0,:,:,sl] - stack_tensor[0,:,:,sl]
            p = outlier_remover(error_tensor)
            likelihood_image[0,:,:,sl] = p

        stack_tensor_corrected = t.mul(stack_tensor,likelihood_image)

        stack["image"] = stack_tensor_corrected
        return stack


    def histogram_normalize_stacks(self, stacks:list)->tuple:
        """        
        Applies histogram normalization to common volume and alls stacks
        Parameters

        Args:
            fixed_image (dict): common/fixed image
            stacks (list): stacks

        Returns:
            tuple: normalized fixed_image, stacks
        """
        normalizer = monai.transforms.HistogramNormalize(
            max=2047, num_bins=2048)
        #fixed_image["image"] = normalizer(fixed_image["image"])
        for st in range(0, self.k):
            stacks[st]["image"] = normalizer(stacks[st]["image"])
        return stacks

    def histogram_normalize(self, fixed_image_tensor:t.tensor)->t.tensor:
        """ 
         normalizes fixed_image_image using Histogram

        Args:
            fixed_image_tensor (t.tensor): tensor to be normalized

        Returns:
            t.tensor: normalized tensor
        """
        normalizer = monai.transforms.HistogramNormalize(
            max=2047, num_bins=2048)
        fixed_image_tensor = normalizer(fixed_image_tensor)
        return fixed_image_tensor

    def normalize(self,stacks:list)->list:
        """
        Normalized to zero mean and std = 1

        Args:
            stacks (list): initial stacks

        Returns:
            list: normalized stacks
        """
        for st in range(0,len(stacks)):
            st_tensor = stacks[st]["image"]
            normalizer = tv.transforms.Normalize(t.mean(st_tensor), t.std(st_tensor))
            stacks[st]["image"] = normalizer(st_tensor)
        return stacks

    def order_stackfilenames_for_preregistration(self, stack_filenames:list)->list:
        """
        return ordered list of stack filenames, with respect to motion corruptions in each stack

        Args:
            stack_filenames (list):

        Returns:
            list: ordered list of filenames
        """
        beta = 0.1

        within_stack_errors = t.zeros(self.k)
        iterations = list()
        for st in range(0,self.k):
            filename = stack_filenames[st]
            path_stack = os.path.join(self.src_folder, filename)
            stack = tio.ScalarImage(path_stack)
            within_stack_error, r = self.within_stack_error(stack.data,beta)
            within_stack_errors[st] = within_stack_error
            iterations.append(r)
        
        _, indices_tensor = t.sort(within_stack_errors)

        #order filenames according to found order
        stack_filenames_ordered = [stack_filenames[i] for i in indices_tensor.tolist()]

        #extract 4 images with least motion corruption
        if len(stack_filenames_ordered) > 4:
            stack_filenames_ordered = stack_filenames_ordered[:4]

        return stack_filenames_ordered

    def within_stack_error(self, img_tensor:t.tensor, beta:int)-> float:
        """Calculates surrogate measure for within stack error caused by motion artefact
        Method from Kainz 2015 eq (2) and eq(3)

        Args:
            img_tensor (t.tensor): data tensor of stac
            beta (int): error_threshold (hyperparameter)

        Returns:
            float: relativ error measure
        """

        img_tensor = img_tensor.squeeze()
        #assumes third dimension to be slice dimension --> flattens the first two dims into 1D vector
        #D: observed data matrix
        D = t.flatten(img_tensor,0,1)

        U,S,V = t.svd(D)
        S = t.diag(S)
        error = beta + 1
        r=0
        while error > beta:
            U_prime = U[:,:r]
            S_prime = S[:r,:r]
            V_prime_t = V[:,:r].transpose(0,1)
            D_prime = t.matmul(t.matmul(U_prime, S_prime),V_prime_t)
            #error measure in equation (2)
            error = t.norm(D-D_prime, p='fro') / t.norm(D, p='fro')
            r +=1
        return (r*error).item(), r