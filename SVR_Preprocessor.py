
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
        self.stack_filenames = stack_filenames
        self.k = len(self.stack_filenames)
        self.mask_filename = mask_filename
        self.mode = monai_mode
        self.tio_mode = tio_mode
        #self.writer = SummaryWriter("runs/test_session")

    def preprocess_stacks_and_common_vol(self, init_pix_dim:tuple, PSF, save_intermediates:bool=False)->tuple:
        """        
        preprocessing procedure before the optimization contains:
        denoising, normalization, initial 3d-3d registration

        Args:
            init_pix_dim (tuple): initial resolution of the fixed image
            save_intermediates (bool, optional):whether to save intermediate steps of preprocessing. Default to False.

        Returns:
            tuple: initial fixed volume, pre registered stacks, slice_dimensions
        """

        #to_device = monai.transforms.ToDeviced(keys = ["image"], device = self.device)
        slice_dimensions = self.crop_images(upsampling=False)
        # load cropped stacks
        stacks = self.load_stacks(to_device=True)
        # denoise stacks
        stacks = self.denoising(stacks)

        if save_intermediates:
            stacks = self.save_stacks(stacks, 'den')

        stacks = self.bias_correction_sitk(stacks)

        if save_intermediates:
            stacks = self.save_stacks(stacks, 'bias')

        stacks = self.normalize(stacks)
        #stacks = self.histogram_normalize_stacks(stacks)
        if save_intermediates:
            stacks = self.save_stacks(stacks, 'norm')

        stacks = self.resample_stacks(stacks, init_pix_dim)

        fixed_image, stacks = self.create_common_volume_registration(stacks, PSF)

        #stacks = self.outlier_removal(fixed_image, stacks)

        fixed_image = self.resample_fixed_image(fixed_image, init_pix_dim)

        if save_intermediates:
            stacks = self.save_stacks(stacks, 'reg')

        for st in range(0, len(stacks)):
            stacks[st]["image"] = stacks[st]["image"].squeeze().unsqueeze(0)

        return fixed_image, stacks, slice_dimensions

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

    def crop_images(self, upsampling:bool=False, pixdim=0)->None:
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

            slice_dimensions.append(list(stack.tensor.shape[1:]).index(min(list(stack.tensor.shape[1:]))))

            resampler = tio.transforms.Resample(stack)
            resampled_mask = resampler(deepcopy(mask))
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

            path_dst = os.path.join(self.prep_folder, filename)
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
            tio_image = self.monai_to_torchio(stacks[st])
            tio_image.save(path)
            image = sitk.ReadImage(path, sitk.sitkFloat32)
            denoised_image = corrector.Execute(image)
            sitk.WriteImage(denoised_image, path)
        
        stacks = self.load_stacks(to_device=True)

        return stacks


    def create_common_volume_registration(self, stacks:list, PSF)->tuple:
        """
        creates common volume and return registered stacks
        Args:
            stacks (list): initial stacks
        Returns:
            tuple: inital fixed image, list of preregistered stacks
        """

        folder = "preprocessing"
        path = os.path.join(folder)
        #to_device = monai.transforms.ToDeviced(keys = ["image"], device = self.device)
        stacks[0]["image"] = stacks[0]["image"].unsqueeze(0)
        fixed_meta = deepcopy(stacks[0]["image_meta_dict"])
        fixed_meta["filename_or_obj"] = "reconstruction_volume.nii.gz"
        common_tensor = stacks[0]["image"].unsqueeze(0)

        affine_transform_monai = monai.networks.layers.AffineTransform(mode = "bilinear",  normalized = True, padding_mode = "zeros")


        tio_common_image = self.monai_to_torchio(stacks[0])
        resample_to_common = tio.transforms.Resample(tio_common_image, image_interpolation=self.tio_mode)

        for st in range(1, self.k):
            stack_tensor = stacks[st]["image"]
            stack_meta = stacks[st]["image_meta_dict"]

            model = custom_models.Volume_to_Volume(PSF, device=self.device)
            loss = loss_module.Loss_Volume_to_Volume("ncc", self.device)
            optimizer = t.optim.Adam(model.parameters(), lr=0.001)

            fixed_image_resampled_tensor = self.resample_fixed_image_to_local_stack(common_tensor,fixed_meta["affine"],stack_tensor,stack_meta["affine"])
            
            stack_tensor = stacks[st]["image"].unsqueeze(0)
            for ep in range(0, 15):
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

            tio_stack = self.monai_to_torchio(stacks[st])
            
            tio_stack = resample_to_common(tio_stack)

            stacks[st] = self.update_monai_from_tio(tio_stack,stacks[st],stacks[st]["image_meta_dict"]["filename_or_obj"])

            common_tensor = common_tensor + stacks[st]["image"]

        normalizer = tv.transforms.Normalize(t.mean(common_tensor), t.std(common_tensor))
        common_tensor = normalizer(common_tensor)
        #common_tensor = t.div(common_tensor, t.max(common_tensor)/2047)

        return {"image": common_tensor.squeeze().unsqueeze(0).unsqueeze(0), "image_meta_dict": fixed_meta}, stacks

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

    def save_stacks(self, stacks:list, post_fix:str)->list:
        """
        saves stack with defined post_fix and retun stacks again, 
        used if save_intermediate during preprocessing is activated

        Args:
            stacks (list): stacks to be save
            post_fix (str): desired postfix on filename

        Returns:
            list: stacks
        """
        folder = "preprocessing"
        path = os.path.join(folder)
        nifti_saver = monai.data.NiftiSaver(output_dir=path, output_postfix=post_fix,
                                            resample=False, mode=self.mode, padding_mode="zeros",
                                            separate_folder=False)

        for st in range(0, len(stacks)):
            nifti_saver.save(stacks[st]["image"].squeeze().unsqueeze(
                0), meta_data=stacks[st]["image_meta_dict"])

        return stacks

    def save_intermediate_reconstruction(self, fixed_image_tensor:t.tensor, fixed_image_meta:dict, epoch:int)->None:
        """
        saves intermediate reconstruction during optimization

        Args:
            fixed_image_tensor (t.tensor): tensor of reconstruction to save
            fixed_image_meta (dict): meta_dict of image to save
            epoch (int): epoch to be added to filename
        """

        path = os.path.join(self.result_folder)
        nifti_saver = monai.data.NiftiSaver(output_dir=path, output_postfix=f"{epoch:02}",
                                            resample=False, mode=self.mode, padding_mode="zeros",
                                            separate_folder=False)
        nifti_saver.save(fixed_image_tensor.squeeze(
        ).unsqueeze(0), meta_data=fixed_image_meta)

    def save_intermediate_reconstruction_and_upsample(self, fixed_image_tensor:t.tensor, fixed_image_meta:dict, epoch:int, upsample:bool = False, pix_dim:tuple = (1,1,1))->dict:
        """
        saves reconstruction of current epoch;
        upsamples fixed image to following resolution for multi resolution approach

        Args:
            fixed_image_tensor (t.tensor): tensor of fixed image
            fixed_image_meta (dict): meta dict of fixed image
            epoch (int): current epoch, is added to filename
            upsample (bool): whether upsampling is necessary
            pix_dim (tuple): pix_dim to sample next

        Returns:
            dict: upsampled monai dict
        """
        self.save_intermediate_reconstruction(fixed_image_tensor, fixed_image_meta, epoch)
        filename = fixed_image_meta["filename_or_obj"]
        filename = filename[:-7] + "_" + f"{epoch:02}" + ".nii.gz"
        #keep filename proper
        filename = filename[:-10] + ".nii.gz"
        tio_image = tio.ScalarImage(tensor=fixed_image_tensor.squeeze().unsqueeze(0).cpu(), affine=fixed_image_meta["affine"])
        if upsample:
            # resample using torchio
            resampler = tio.transforms.Resample(pix_dim, image_interpolation=self.tio_mode)
            tio_resampled = resampler(tio_image)
            monai_resampled = self.update_monai_from_tio(tio_resampled,{"image":fixed_image_tensor, "image_meta_dict": fixed_image_meta}, filename)
        else:
            monai_resampled = self.update_monai_from_tio(tio_image,{"image":fixed_image_tensor, "image_meta_dict": fixed_image_meta}, filename)

        return monai_resampled

    def resample_fixed_image(self, fixed_image:dict, pix_dim:tuple)->dict:
        """
        Can be used to resample fixed image without saving it

        Args:
            fixed_image (dict): monai dict
            pix_dim (tuple): desired pixdim/resolution

        Returns:
            dict: upsampled monai dict
        """
        filename = fixed_image["image_meta_dict"]["filename_or_obj"]
        path = os.path.join(self.result_folder)
        nifti_saver = monai.data.NiftiSaver(output_dir=path, output_postfix=f"{-1:02}",
                                            resample=False, mode=self.mode, padding_mode="zeros",
                                            separate_folder=False)
        nifti_saver.save(fixed_image["image"].squeeze().unsqueeze(
            0), meta_data=fixed_image["image_meta_dict"])

        fixed_image["image_meta_dict"]["filename_or_obj"] = filename[:-10] + ".nii.gz"
        filename = filename[:-7] + "_" + f"{-1:02}" + ".nii.gz"

        path = os.path.join(self.result_folder, filename)
        fixed_image_tio = self.monai_to_torchio(fixed_image)
        resampler = tio.transforms.Resample(pix_dim, image_interpolation=self.tio_mode)
        resampled_fixed_tio = resampler(fixed_image_tio)
        fixed_image = self.update_monai_from_tio(resampled_fixed_tio,fixed_image,filename[:-10] + ".nii.gz")

        return fixed_image

    def resample_stacks(self, stacks:list, pix_dim:tuple)->list:

        stacks_updated = list()
        resampler = tio.transforms.Resample(pix_dim, image_interpolation=self.tio_mode)
        for st in range(0,len(stacks)):
            tio_stack = self.monai_to_torchio(stacks[st])
            tio_stack_resampled = resampler(tio_stack)
            filename = stacks[st]["image_meta_dict"]["filename_or_obj"]
            resampled_monai = self.update_monai_from_tio(tio_stack_resampled,stacks[st],filename)
            resampled_monai["image"] = resampled_monai["image"].squeeze().unsqueeze(0)
            stacks_updated.append(resampled_monai)
        
        return stacks_updated


    def monai_to_torchio(self, monai_dict:dict)->tio.ScalarImage:
        """
        takes monai dict and return corresponding tio Image, output is on cpu!!

        Args:
            monai_dict (dict):

        Returns:
            tio.ScalarImage:
        """
        return tio.ScalarImage(tensor=monai_dict["image"].squeeze().unsqueeze(0).detach().cpu(), affine=monai_dict["image_meta_dict"]["affine"])
    
    def update_monai_from_tio(self, tio_image:tio.ScalarImage, monai_dict:dict, filename:str) -> dict:
        """
        updated monai dict from given tio Image, puts image back on current device (possibly gpu)

        Args:
            tio_image (tio.Image):
            monai_dict (dict): initial monai dict
            filename (str): filename to be updated in meta dict

        Returns:
            dict: updated monai dict
        """
        to_device = monai.transforms.ToDeviced(keys = ["image"], device = self.device)
        add_channel = AddChanneld(keys=["image"])


        monai_dict["image"] = tio_image.tensor.to(self.device)
        monai_dict["image_meta_dict"]["affine"] = tio_image.affine
        monai_dict["image_meta_dict"]["spatial_shape"] = np.array(list(tio_image.tensor.shape)[1:])
        monai_dict["image_meta_dict"]["filename_or_obj"] = filename

        monai_dict = to_device(monai_dict)
        monai_dict = add_channel(monai_dict)

        return monai_dict


    def resample_fixed_image_to_local_stack(self, fixed_image_tensor:t.tensor, fixed_image_affine:t.tensor, local_stack_tensor:t.tensor, local_stack_affine:t.tensor)->t.tensor:
        """
        Args:
            fixed_image_tensor (t.tensor): 
            fixed_image_affine (t.tensor): 
            local_stack_tensor (t.tensor): 
            local_stack_affine (t.tensor): 
        Returns:
            t.tensor: fixed_image_tensor resmpled to local stack
        """
        local_stack_tio = tio.Image(tensor=local_stack_tensor, affine = local_stack_affine)
        resampler_tio = tio.transforms.Resample(local_stack_tio, image_interpolation= self.tio_mode)
        #resample fixed image to loca stack
        fixed_tio = tio.Image(tensor=fixed_image_tensor.squeeze().unsqueeze(0).detach().cpu(), affine=fixed_image_affine) 
        fixed_tio = resampler_tio(fixed_tio)
        fixed_image_tensor = fixed_tio.tensor.to(self.device)
        return fixed_image_tensor