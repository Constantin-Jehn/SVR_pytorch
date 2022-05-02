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
import SimpleITK as sitk


class Preprocesser():
    def __init__(self, src_folder, prep_folder, result_folder, stack_filenames, mask_filename, device, mode):
        self.device = device
        self.src_folder = src_folder
        self.prep_folder = prep_folder
        self.result_folder = result_folder
        self.stack_filenames = stack_filenames
        self.k = len(self.stack_filenames)
        self.mask_filename = mask_filename
        self.mode = mode

    def preprocess_stacks_and_common_vol(self, init_pix_dim, save_intermediates=False):
        """
        Parameters
        ----------
        init_pix_dim : tuple
            initial resolution of the fixed image
        save_intermediates : boolean
            whether to save intermediat steps of preprocessing
        Returns
        -------
        fixed_images: dict
            inital common volume
        stacks: list
            preprocessed stacks
        """
        #to_device = monai.transforms.ToDeviced(keys = ["image"], device = self.device)
        self.crop_images(upsampling=False)
        # load cropped stacks
        stacks = self.load_stacks(to_device=True)
        # denoise stacks
        stacks = self.denoising(stacks)

        if save_intermediates:
            stacks = self.save_stacks(stacks, 'den')
        # self.bias_correction_sitk(stacks)
        fixed_images, stacks = self.create_common_volume_registration(stacks)

        fixed_images = self.resample_fixed_image(fixed_images, init_pix_dim)

        if save_intermediates:
            stacks = self.save_stacks(stacks, 'reg')

        fixed_images, stacks = self.histogram_normalize(fixed_images, stacks)
        if save_intermediates:
            stacks = self.save_stacks(stacks, 'norm')
        # self.fixed_images = add_channel(self.fixed_images)
        for st in range(0, len(stacks)):
            stacks[st]["image"] = stacks[st]["image"].squeeze().unsqueeze(0)
        #fixed_images["image"] = fixed_images["image"].squeeze().unsqueeze(0)
        return fixed_images, stacks

    def get_cropped_stacks(self):
        self.crop_images(upsampling=False)
        # load cropped stacks
        stacks = self.load_stacks(to_device=True)
        return stacks

    def crop_images(self, upsampling=False, pixdim=0):
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

        for i in range(0, self.k):
            filename = self.stack_filenames[i]
            path_stack = os.path.join(self.src_folder, filename)
            stack = tio.ScalarImage(path_stack)
            resampler = tio.transforms.Resample(stack)
            resampled_mask = resampler(deepcopy(mask))
            subject = tio.Subject(stack=stack, mask=resampled_mask)

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
                    upsampler = tio.transforms.Resample(pixdim)
                    cropped_stack = upsampler(cropped_stack)
                else:
                    path_stack = os.path.join(
                        self.prep_folder, self.stack_filenames[0])
                    resampler = tio.transforms.Resample(path_stack)
                    cropped_stack = resampler(cropped_stack)

            path_dst = os.path.join(self.prep_folder, filename)
            cropped_stack.stack.save(path_dst)

    def load_stacks(self, to_device=False):
        """
        After cropping the initial images in low resolution are saved in their original coordinates
        for the loss computation
        Returns
        -------
        ground_truth : list
            list of dictionaries containing the ground truths

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

    def denoising(self, stacks):
        """
        Applies Gaussian Sharpen Filter to all stacks

        Parameters
        ----------
        stacks : list

        Returns
        -------
        stacks : list
        """
        gauss_sharpen = monai.transforms.GaussianSharpen(
            sigma1=1, sigma2=1, alpha=3)
        for st in range(0, self.k):
            stacks[st]["image"] = gauss_sharpen(stacks[st]["image"])
        return stacks

    def denoise_single_slice(self, single_slice_image):
        gauss_sharpen = monai.transforms.GaussianSharpen(
            sigma1=1, sigma2=1, alpha=3)
        return gauss_sharpen(single_slice_image)

    def bias_correction_sitk(self, stacks):
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        for st in range(0, self.k):
            path = os.path.join(
                stacks[st]["image_meta_dict"]["filename_or_obj"])
            image = sitk.ReadImage(path, sitk.sitkFloat32)
            denoised_image = corrector.Execute(image)
            path = os.path.join(self.prep_folder, self.stack_filenames[st])
            sitk.WriteImage(denoised_image, path)

    def create_common_volume_registration(self, stacks):
        """
        creates common volume and return registered stacks

        Parameters
        ----------
        stacks : list

        Returns
        -------
        dict
            common volume
        stacks : list
            registered stacks

        """
        folder = "preprocessing"
        path = os.path.join(folder)
        #to_device = monai.transforms.ToDeviced(keys = ["image"], device = self.device)
        stacks[0]["image"] = stacks[0]["image"].unsqueeze(0)
        fixed_meta = stacks[0]["image_meta_dict"]
        fixed_meta["filename_or_obj"] = "reconstruction_volume.nii.gz"
        common_image = stacks[0]["image"].unsqueeze(0)

        for st in range(1, self.k):
            image = stacks[st]["image"].unsqueeze(0)
            meta = stacks[st]["image_meta_dict"]

            model = custom_models.Reconstruction(
                n_slices=1, device=self.device)
            loss = loss_module.RegistrationLoss("ncc", self.device)
            optimizer = t.optim.Adam(model.parameters(), lr=0.001)

            for ep in range(0, 15):
                transformed = model(image.detach(), meta,
                                    fixed_meta, transform_to_fixed=True)
                transformed = transformed.to(self.device)
                loss_tensor = loss(transformed, stacks[0])
                loss_tensor.backward()
                if ep < 14:
                    optimizer.step()

            transformed = transformed.detach()

            common_image = common_image + transformed

            #stacks[st]["image"],stacks[st]["image_meta_dict"]["affine"], stacks[st]["image_meta_dict"]["spatial_shape"] = transformed, fixed_meta["affine"], fixed_meta["spatial_shape"]
            pre_registered = model(stacks[st]["image"].unsqueeze(
                0), meta, fixed_meta, transform_to_fixed=False)
            stacks[st]["image"] = pre_registered

        return {"image": common_image.squeeze().unsqueeze(0).unsqueeze(0), "image_meta_dict": fixed_meta}, stacks

    def histogram_normalize(self, fixed_images, stacks):
        """
        Applies histogram normalization to common volume and alls stacks
        Parameters
        ----------
        fixed_images : dict
            common_volume
        stacks : list
            DESCRIPTION.

        Returns
        -------
        fixed_images : dict

        stacks : list
        """
        add_channel = AddChanneld(keys=["image"])
        loader = LoadImaged(keys=["image"])
        to_tensor = ToTensord(keys=["image"])
        resampler = monai.transforms.ResampleToMatch(mode=self.mode)
        normalizer = monai.transforms.HistogramNormalize(
            max=2047, num_bins=2048)

        path_mask = os.path.join(self.src_folder, self.mask_filename)
        mask_dict = {"image": path_mask}
        mask = loader(mask_dict)

        mask = to_tensor(mask)
        mask = add_channel(mask)

        mask["image"], mask["image_meta_dict"] = resampler(mask["image"], src_meta=mask["image_meta_dict"],
                                                           dst_meta=fixed_images["image_meta_dict"])
        mask["image_meta_dict"]["spatial_shape"] = np.array(
            list(mask["image"].shape)[1:])

        mask = add_channel(mask)

        fixed_images["image"] = normalizer(fixed_images["image"])

        mask["image"] = mask["image"].squeeze().unsqueeze(0)

        for st in range(0, self.k):
            mask["image"], mask["image_meta_dict"] = resampler(mask["image"], src_meta=mask["image_meta_dict"],
                                                               dst_meta=stacks[st]["image_meta_dict"])
            mask["image_meta_dict"]["spatial_shape"] = np.array(
                list(mask["image"].shape)[1:])

            mask = add_channel(mask)

            stacks[st]["image"] = normalizer(
                stacks[st]["image"], mask["image"])

            mask["image"] = mask["image"].squeeze().unsqueeze(0)

        return fixed_images, stacks

    def normalize(self, fixed_image_image):
        """
        normalizes fixed_image_image using Histogram
        """
        normalizer = monai.transforms.HistogramNormalize(
            max=2047, num_bins=2048)
        fixed_image_image = normalizer(fixed_image_image)
        return fixed_image_image

    def save_stacks(self, stacks, post_fix):
        folder = "preprocessing"
        path = os.path.join(folder)
        nifti_saver = monai.data.NiftiSaver(output_dir=path, output_postfix=post_fix,
                                            resample=False, mode=self.mode, padding_mode="zeros",
                                            separate_folder=False)

        for st in range(0, len(stacks)):
            nifti_saver.save(stacks[st]["image"].squeeze().unsqueeze(
                0), meta_data=stacks[st]["image_meta_dict"])

        return stacks

    def save_intermediate_reconstruction(self, fixed_image_image, fixed_image_meta, epoch):

        path = os.path.join(self.result_folder)
        nifti_saver = monai.data.NiftiSaver(output_dir=path, output_postfix=f"{epoch:02}",
                                            resample=False, mode=self.mode, padding_mode="zeros",
                                            separate_folder=False)
        nifti_saver.save(fixed_image_image.squeeze(
        ).unsqueeze(0), meta_data=fixed_image_meta)

    def save_intermediate_reconstruction_and_upsample(self, fixed_image_image, fixed_image_meta, epoch, pix_dim):
        self.save_intermediate_reconstruction(
            fixed_image_image, fixed_image_meta, epoch)
        # resample using torchio
        filename = fixed_image_meta["filename_or_obj"]
        filename = filename[:-7] + "_" + f"{epoch:02}" + ".nii.gz"
        path = os.path.join(self.result_folder, filename)
        fixed_image = tio.ScalarImage(path)
        resampler = tio.transforms.Resample(pix_dim)
        resamped_fixed = resampler(fixed_image)
        resamped_fixed.save(path)
        # load as nifti file and return
        add_channel = AddChanneld(keys=["image"])
        loader = LoadImaged(keys=["image"])
        to_tensor = ToTensord(keys=["image"])
        to_device = monai.transforms.ToDeviced(
            keys=["image"], device=self.device)
        fixed_dict = {"image": path}
        fixed_dict = loader(fixed_dict)
        fixed_dict = to_tensor(fixed_dict)
        fixed_dict = add_channel(fixed_dict)
        # keep meta data correct
        fixed_dict["image_meta_dict"]["spatial_shape"] = np.array(
            list(fixed_dict["image"].shape)[1:])
        fixed_dict["image_meta_dict"]["filename_or_obj"] = filename[:-10] + ".nii.gz"
        # move to gpu
        fixed_dict = to_device(fixed_dict)
        return fixed_dict

    def resample_fixed_image(self, fixed_image, pix_dim):
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
        fixed_image = tio.ScalarImage(path)

        resampler = tio.transforms.Resample(pix_dim)
        resamped_fixed = resampler(fixed_image)

        resamped_fixed.save(path)
        # load as nifti file and return

        add_channel = AddChanneld(keys=["image"])
        loader = LoadImaged(keys=["image"])
        to_tensor = ToTensord(keys=["image"])
        to_device = monai.transforms.ToDeviced(
            keys=["image"], device=self.device)
        fixed_dict = {"image": path}
        fixed_dict = loader(fixed_dict)
        fixed_dict = to_tensor(fixed_dict)
        fixed_dict = add_channel(fixed_dict)
        fixed_dict = add_channel(fixed_dict)
        # keep meta data correct
        fixed_dict["image_meta_dict"]["spatial_shape"] = np.array(
            list(fixed_dict["image"].shape)[2:])
        fixed_dict["image_meta_dict"]["filename_or_obj"] = filename[:-10] + ".nii.gz"
        # move to gpu
        fixed_dict = to_device(fixed_dict)

        return fixed_dict
