import monai
from monai.transforms import (
    AddChanneld
)
import torchio as tio
import os
import torch as t
import numpy as np

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

def save_stacks(stacks:list, post_fix:str, mode:str)->list:
    """
    saves stack with defined post_fix and retun stacks again, 
    used if save_intermediate during preprocessing is activated

    Args:
        stacks (list): stacks to be save
        post_fix (str): desired postfix on filename
        mode (str): monai mode for saving

    Returns:
        list: stacks
    """
    folder = "preprocessing"
    path = os.path.join(folder)
    nifti_saver = monai.data.NiftiSaver(output_dir=path, output_postfix=post_fix,
                                        resample=False, mode=mode, padding_mode="zeros",
                                        separate_folder=False)

    for st in range(0, len(stacks)):
        nifti_saver.save(stacks[st]["image"].squeeze().unsqueeze(
            0), meta_data=stacks[st]["image_meta_dict"])

    return stacks

def save_intermediate_reconstruction(fixed_image_tensor:t.tensor, fixed_image_meta:dict, epoch:int, result_folder:str, mode:str)->None:
    """
    saves intermediate reconstruction during optimization

    Args:
        fixed_image_tensor (t.tensor): tensor of reconstruction to save
        fixed_image_meta (dict): meta_dict of image to save
        epoch (int): epoch to be added to filename
        result_folder(str): where to store results
        mode(str): mode for saving
    """

    path = os.path.join(result_folder)
    nifti_saver = monai.data.NiftiSaver(output_dir=path, output_postfix=f"{epoch:02}",
                                        resample=False, mode=mode, padding_mode="zeros",
                                        separate_folder=False)
    nifti_saver.save(fixed_image_tensor.squeeze(
    ).unsqueeze(0), meta_data=fixed_image_meta)


def save_intermediate_reconstruction_and_upsample(fixed_image_tensor:t.tensor, fixed_image_meta:dict, epoch:int, result_folder:str, mode:str, tio_mode:str, upsample:bool = False, pix_dim:tuple = (1,1,1), )->dict:
        """
        saves reconstruction of current epoch;
        upsamples fixed image to following resolution for multi resolution approach

        Args:
            fixed_image_tensor (t.tensor): tensor of fixed image
            fixed_image_meta (dict): meta dict of fixed image
            epoch (int): current epoch, is added to filename
            result_folder(str)
            mode(str):monai mode
            tio_mode(str): torchio mode
            upsample (bool): whether upsampling is necessary
            pix_dim (tuple): pix_dim to sample next

        Returns:
            dict: upsampled monai dict
        """
        save_intermediate_reconstruction(fixed_image_tensor, fixed_image_meta, epoch, result_folder, mode)
        filename = fixed_image_meta["filename_or_obj"]
        filename = filename[:-7] + "_" + f"{epoch:02}" + ".nii.gz"
        #keep filename proper
        filename = filename[:-10] + ".nii.gz"
        tio_image = tio.ScalarImage(tensor=fixed_image_tensor.squeeze().unsqueeze(0).cpu(), affine=fixed_image_meta["affine"])
        if upsample:
            # resample using torchio
            resampler = tio.transforms.Resample(pix_dim, image_interpolation=tio_mode)
            tio_resampled = resampler(tio_image)
            monai_resampled = update_monai_from_tio(tio_resampled,{"image":fixed_image_tensor, "image_meta_dict": fixed_image_meta}, filename)
        else:
            monai_resampled = update_monai_from_tio(tio_image,{"image":fixed_image_tensor, "image_meta_dict": fixed_image_meta}, filename)

        return monai_resampled

def resample_fixed_image(fixed_image:dict, pix_dim:tuple, result_folder:str, mode:str, tio_mode:str)->dict:
    """
    Can be used to resample fixed image and saving it as initial image (-1)

    Args:
        fixed_image (dict): monai dict
        pix_dim (tuple): desired pixdim/resolution
        result_folder(str)
        mode(str):monai mode
        tio_mode(str): torchio mode

    Returns:
        dict: upsampled monai dict
    """
    filename = fixed_image["image_meta_dict"]["filename_or_obj"]
    path = os.path.join(result_folder)
    nifti_saver = monai.data.NiftiSaver(output_dir=path, output_postfix=f"{-1:02}",
                                        resample=False, mode=mode, padding_mode="zeros",
                                        separate_folder=False)
    nifti_saver.save(fixed_image["image"].squeeze().unsqueeze(
        0), meta_data=fixed_image["image_meta_dict"])

    fixed_image["image_meta_dict"]["filename_or_obj"] = filename[:-10] + ".nii.gz"
    filename = filename[:-7] + "_" + f"{-1:02}" + ".nii.gz"

    path = os.path.join(result_folder, filename)
    fixed_image_tio = monai_to_torchio(fixed_image)
    resampler = tio.transforms.Resample(pix_dim, image_interpolation=tio_mode)
    resampled_fixed_tio = resampler(fixed_image_tio)
    fixed_image = update_monai_from_tio(resampled_fixed_tio,fixed_image,filename[:-10] + ".nii.gz")

    return fixed_image

def resample_stacks(stacks:list, pix_dim:tuple, tio_mode:str)->list:
    """resamples a list of stacks to given pis_dim

    Args:
        stacks (list): list of stacks as monai dicts
        pix_dim (tuple): desired pix dim
        tio_mode (str): resmpling mode for torchio

    Returns:
        list: updated monai stacks
    """

    stacks_updated = list()
    resampler = tio.transforms.Resample(pix_dim, image_interpolation=tio_mode)
    for st in range(0,len(stacks)):
        tio_stack = monai_to_torchio(stacks[st])
        tio_stack_resampled = resampler(tio_stack)
        filename = stacks[st]["image_meta_dict"]["filename_or_obj"]
        resampled_monai = update_monai_from_tio(tio_stack_resampled,stacks[st],filename)
        resampled_monai["image"] = resampled_monai["image"].squeeze().unsqueeze(0)
        stacks_updated.append(resampled_monai)
    
    return stacks_updated


def monai_to_torchio(monai_dict:dict)->tio.ScalarImage:
    """
    takes monai dict and return corresponding tio Image, output is on cpu!!

    Args:
        monai_dict (dict):

    Returns:
        tio.ScalarImage:
    """
    return tio.ScalarImage(tensor=monai_dict["image"].squeeze().unsqueeze(0).detach().cpu(), affine=monai_dict["image_meta_dict"]["affine"])

def update_monai_from_tio(tio_image:tio.ScalarImage, monai_dict:dict, filename:str, device:str = device) -> dict:
    """
    updated monai dict from given tio Image, puts image back on current device (possibly gpu)

    Args:
        tio_image (tio.Image):
        monai_dict (dict): initial monai dict
        filename (str): filename to be updated in meta dict
        device (str): cuda or cpu

    Returns:
        dict: updated monai dict
    """
    to_device = monai.transforms.ToDeviced(keys = ["image"], device = device)
    add_channel = AddChanneld(keys=["image"])


    monai_dict["image"] = tio_image.tensor.to(device)
    monai_dict["image_meta_dict"]["affine"] = tio_image.affine
    monai_dict["image_meta_dict"]["spatial_shape"] = np.array(list(tio_image.tensor.shape)[1:])
    monai_dict["image_meta_dict"]["filename_or_obj"] = filename

    monai_dict = to_device(monai_dict)
    monai_dict = add_channel(monai_dict)

    return monai_dict

def resample_fixed_image_to_local_stack(fixed_image_tensor:t.tensor, fixed_image_affine:t.tensor, local_stack_tensor:t.tensor, local_stack_affine:t.tensor, tio_mode:str, device:str)->t.tensor:
    """
    Args:
        fixed_image_tensor (t.tensor): 
        fixed_image_affine (t.tensor): 
        local_stack_tensor (t.tensor): 
        local_stack_affine (t.tensor): 
        tio_mode(str)
        device(str)
    Returns:
        t.tensor: fixed_image_tensor resmpled to local stack
    """
    local_stack_tensor_cpu = local_stack_tensor.detach().cpu()
    local_stack_tio = tio.Image(tensor=local_stack_tensor_cpu, affine = local_stack_affine)
    resampler_tio = tio.transforms.Resample(local_stack_tio, image_interpolation= tio_mode)
    #resample fixed image to loca stack
    tensor_cpu = fixed_image_tensor.squeeze().unsqueeze(0).cpu()
    affine_cpu = fixed_image_affine
    fixed_tio = tio.Image(tensor=tensor_cpu, affine=affine_cpu) 
    fixed_tio = resampler_tio(fixed_tio)
    fixed_image_tensor = fixed_tio.tensor.to(device)
    return fixed_image_tensor
