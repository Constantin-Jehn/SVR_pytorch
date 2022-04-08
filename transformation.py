import matplotlib.pyplot as plt
import monai
import torchio as tio
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
import utils
import numpy as np
import reconstruction_model
import torch as t
import monai.transforms.utils as monai_utils
from copy import deepcopy
import nibabel as nib

def crop_images(src_folder, filenames, mask_filename, dst_folder):
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

    Returns
    -------
    None.
    """
    path_mask = os.path.join(src_folder, mask_filename)
    mask = tio.LabelMap(path_mask)
    path_dst = os.path.join(dst_folder, mask_filename)
    mask.save(path_dst)
    k = len(filenames)
    for i in range(0,k):
        filename = filenames[i]
        path_stack = os.path.join(src_folder, filename)
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
        path_dst = os.path.join(dst_folder, filename)
        cropped_stack.stack.save(path_dst)
        
    
def load_files(folder, filenames, mask_filename):
    """
    loads stacks, and mask from specified filenames
    Parameters
    ----------
    folder : string
        folder name must be in current directory
    filenames : list of string
        filenames of nifti files of stacks
    mask_filename : string
        nifti filename of mask

    Returns
    -------
    stacks : list
        contains each stack as dictionary with image and meta data
    mask : dictionary
        contains mask with image and meta data

    """
    #necessary transforms
    add_channel = AddChanneld(keys=["image"])
    to_tensor = ToTensord(keys = ("image"))
    orientation = monai.transforms.Orientation(axcodes="PLI", image_only=False)
    loader = LoadImaged(keys = ("image"))
    
    k = len(filenames)
    stacks = list()
    for file in range(0,k):
        path = os.path.join(folder, filenames[file])
        nifti_dict = loader({"image": path})
        nifti_dict = to_tensor(nifti_dict)
        nifti_dict = add_channel(nifti_dict)
        
        #nifti_dict["image"], _ ,  nifti_dict["image_meta_dict"]["affine"] = orientation(nifti_dict["image"], affine = nifti_dict["image_meta_dict"]["affine"])
        #nifti_dict["image_meta_dict"]["spatial_shape"] = list(nifti_dict["image"].shape[1:])
        stacks.append(nifti_dict)
    
    path = os.path.join(folder, mask_filename)
    nifti_dict = loader({"image": path})
    nifti_dict = to_tensor(nifti_dict)
    nifti_dict = add_channel(nifti_dict)
    
    
    #nifti_dict["image"], _ ,  nifti_dict["image_meta_dict"]["affine"] = orientation(nifti_dict["image"], affine = nifti_dict["image_meta_dict"]["affine"])
    #nifti_dict["image_meta_dict"]["spatial_shape"] = list(nifti_dict["image"].shape[1:])
    mask = nifti_dict
    
    return stacks, mask
        
def resample_to_common_coord(stacks, mask):
    """
    Resamples different stacks to one common coordinate system
    Parameters
    ----------
    stacks : list
        list of stacks as dictionaries
    mask : dictionary
        mask as dictionary

    Returns
    -------
    stacks : list
        all stacks in one coordinate system (chosen from one of the stacks)
    mask : dictionary
        mask in common coordinate system for cropping
    """
    interpolation_mode = "bilinear"
    padding_mode = "zeros"
    resampler = monai.transforms.ResampleToMatch()
    k = len(stacks)
    depth = 500
    ind_min_depth = -1
    for st in range(0,k):
        if stacks[st]["image"].shape[-1] < depth:
            depth = stacks[st]["image"].shape[-1]
            ind_min_depth = st
        
    
    for st in range(0,k):
        #keep filename for saving later on
        file_obj = deepcopy(stacks[st]["image_meta_dict"]["filename_or_obj"])
        original_affine = deepcopy(stacks[st]["image_meta_dict"]["original_affine"])
        stacks[st]["image"], stacks[st]["image_meta_dict"] = resampler(stacks[st]["image"],src_meta = stacks[st]["image_meta_dict"], dst_meta = stacks[ind_min_depth]["image_meta_dict"],
                                                                       mode = interpolation_mode, padding_mode = padding_mode)
        stacks[st]["image_meta_dict"]["filename_or_obj"] = file_obj
        stacks[st]["image_meta_dict"]["original_affine"] = original_affine
        
    file_obj = deepcopy(mask["image_meta_dict"]["filename_or_obj"])
    original_affine = deepcopy(mask["image_meta_dict"]["original_affine"])
    mask["image"], mask["image_meta_dict"] = resampler(mask["image"],src_meta = mask["image_meta_dict"], dst_meta = stacks[ind_min_depth]["image_meta_dict"],
                                                       mode = interpolation_mode, padding_mode = padding_mode)
    mask["image_meta_dict"]["filename_or_obj"] = file_obj
    mask["image_meta_dict"]["original_affine"] = original_affine
    return stacks, mask
 

def resample_to_original_coord(stacks):
    interpolation_mode = "bilinear"
    padding_mode = "zeros"
    resampler = monai.transforms.SpatialResample()
    k = len(stacks)
    for st in range(0,k):
        stacks[st]["image"], stacks[st]["image_meta_dict"]["affine"] = resampler(stacks[st]["image"],src_affine = stacks[st]["image_meta_dict"]["affine"], dst_affine = stacks[st]["image_meta_dict"]["original_affine"],
                                                                       mode = interpolation_mode, padding_mode = padding_mode)
    return stacks


def resample_to_pixdim(stacks, mask, pixdim):
    """
    resamples stacks and mask (in common coordinate system) to given pixdim (or voxel dimension)
    Parameters
    ----------
    stacks : list
        list of stacks in common coordinate system
    mask : dictionary
        mask
    pixdim : Sequence(float)
        desired pixel/voxel dimensions

    Returns
    -------
    stacks : list
        list of resamples stacks
    mask : dictionary
        resampled mask

    """
    interpolation_mode = "bilinear"
    padding_mode = "zeros"
    spacing = Spacingd(keys = ["image"], pixdim = pixdim, mode = interpolation_mode, padding_mode = padding_mode)
    k = len(stacks)
    for st in range(0,k):
        stacks[st] = spacing(stacks[st])
    mask = spacing(mask)
    return stacks, mask

def save_to_nifti(stacks, mask):
    """
    save list of stacks and mask to nifti format
    Parameters
    ----------
    stacks : list
        list of stack to save to nifti format
    mask : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    """
    folder = "test_reconstruction_monai"
    folder2 = "stacks"
    path = os.path.join(folder,folder2)
    
    mode = "bilinear"
    nifti_saver = monai.data.NiftiSaver(output_dir=path, 
                                        resample = False, mode = mode, padding_mode = "zeros",
                                        separate_folder=False)
    k = len(stacks)
    for st in range(0,k):
        nifti_saver.save(stacks[st]["image"], meta_data=stacks[st]["image_meta_dict"])
    nifti_saver.save(mask["image"], meta_data=mask["image_meta_dict"])        


filenames = ["10_3T_nody_001.nii.gz",
"10_3T_nody_002.nii.gz",
"14_3T_nody_001.nii.gz",
"14_3T_nody_002.nii.gz",
"21_3T_nody_001.nii.gz",
"21_3T_nody_002.nii.gz",
"23_3T_nody_001.nii.gz",
"23_3T_nody_002.nii.gz"]
file_mask = "mask_10_3T_brain_smooth.nii.gz"
pixdim = (1.0, 1.0, 1.0)

src_folder = "sample_data"
dst_folder = "cropped_images"


crop_images(src_folder, filenames, file_mask, dst_folder)

stacks, mask = load_files(dst_folder, filenames, file_mask)

stacks, mask = resample_to_common_coord(stacks, mask)

stacks = resample_to_original_coord(stacks)

save_to_nifti(stacks, mask)



#stacks, mask = crop_brain_individually(stacks,mask)

# grid = monai.transforms.utils.create_grid([100,100,100])

# test_stack = stacks[0]

# resampler = monai.transforms.Resample()
# resampled_image = resampler(test_stack["image"], grid)



#stacks, mask = resample_to_pixdim(stacks, mask, pixdim)

#stacks, mask = crop_brain(stacks, mask)
#stacks = resample_to_original_coord(stacks)


#stacks, mask = resample_to_pixdim(stacks, mask, pixdim)




# resampler = monai.transforms.ResampleToMatch()
# spat_resampler = monai.transforms.SpatialResampled(keys=["image"], meta_keys=["image_meta_dict"], meta_src_keys=["affine"], meta_dst_keys=["dst_affine"])

# affine_1 = monai.transforms.Affine(affine = np.linalg.inv(target_dict_1["image_meta_dict"]["affine"]))
# affine_2 = monai.transforms.Affine(affine = np.linalg.inv(target_dict_2["image_meta_dict"]["affine"]))
# affine_mask = monai.transforms.Affine(affine = np.linalg.inv(mask_dict["image_meta_dict"]["affine"]))

#spatial_size = (500,500,500)
# affine_1 = monai.transforms.Affine(affine = np.linalg.inv(target_dict_1["image_meta_dict"]["affine"]), norm_coords= True)
# affine_2 = monai.transforms.Affine(affine = np.linalg.inv(target_dict_2["image_meta_dict"]["affine"]),norm_coords= True)
# affine_mask = monai.transforms.Affine(affine =np.linalg.inv( mask_dict["image_meta_dict"]["affine"]), norm_coords= True)


# spatial_res = monai.transforms.SpatialResample()
# pixdim = [0,1,1,1,0,0,0,0]
# world_spatial = (300,300,300)
# world_affine = t.eye(4)
# world_affine[3,3] = 1

# world_meta = {"affine":world_affine, "pixdim": pixdim}

# target_dict_1["image_meta_dict"].update({"dst_affine": world_affine})
# target_dict_2["image_meta_dict"].update({"dst_affine": world_affine})
# mask_dict["image_meta_dict"].update({"dst_affine": world_affine})

#apply inv affine
# target_dict_1["image"], target_dict_1["image_meta_dict"]["affine"] = affine_1(target_dict_1["image"])
# target_dict_2["image"], target_dict_2["image_meta_dict"]["affine"] = affine_2(target_dict_2["image"])
# mask_dict["image"], mask_dict["image_meta_dict"]["affine"] = affine_mask(mask_dict["image"])

#resample to match
# target_dict_1["image"], target_dict_1["image_meta_dict"] = resampler(target_dict_1["image"], src_meta = target_dict_1["image_meta_dict"], dst_meta = world_meta)
# target_dict_2["image"], target_dict_2["image_meta_dict"] = resampler(target_dict_2["image"], src_meta = target_dict_2["image_meta_dict"], dst_meta = world_meta)
# mask_dict["image"], mask_dict["image_meta_dict"] = resampler(mask_dict["image"], src_meta = mask_dict["image_meta_dict"], dst_meta = world_meta)


# target_dict_1 = spacing(target_dict_1)
# target_dict_2 = spacing(target_dict_2)
# mask_dict = spacing(mask_dict)

# target_dict_1 = spat_resampler(target_dict_1)
# target_dict_2 = spat_resampler(target_dict_2)
# mask_dict = spat_resampler(mask_dict)




# grid = monai_utils.create_grid([200,100,100],backend = "torch")
# ext_1 = monai_utils.get_extreme_points(target_dict_1["image"])

# target_dict_2["image"], target_dict_2["image_meta_dict"] = resampler(target_dict_2["image"], src_meta = target_dict_2["image_meta_dict"], dst_meta = target_dict_1["image_meta_dict"])
# # target_dict_1["image"], target_dict_1["image_meta_dict"] = resampler(target_dict_1["image"], src_meta=target_dict_1["image_meta_dict"], dst_meta=target_1_mask["image_meta_dict"])



#make first dimension the slices
#target_dict = orientation(target_dict)