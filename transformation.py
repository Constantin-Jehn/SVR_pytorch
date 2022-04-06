import matplotlib.pyplot as plt
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
import utils
import numpy as np
import reconstruction_model
import torch as t
import monai.transforms.utils as monai_utils
from copy import deepcopy


def load_files(folder, filenames):
    pixdim = (3,3,3)
    mode = "bilinear"
    #necessary transforms
    add_channel = AddChanneld(keys=["image"])
    orientation = monai.transforms.Orientationd(keys = ("image"), axcodes="PLI")
    to_tensor = ToTensord(keys = ("image"))
    loader = LoadImaged(keys = ("image"))
    
    k = len(filenames)
    stacks = list()
    for file in range(0,k):
        path = os.path.join(folder, filenames[file])
        nifti_dict = loader({"image": path})
        nifti_dict = to_tensor(nifti_dict)
        nifti_dict = add_channel(nifti_dict)
        stacks.append(nifti_dict)
    return stacks
        
def resample_to_world_coord(stacks):
    resampler = monai.transforms.ResampleToMatch()
    k = len(stacks)
    for st in range(1,k):
        file_obj = deepcopy(stacks[st]["image_meta_dict"]["filename_or_obj"])
        stacks[st]["image"], stacks[st]["image_meta_dict"] = resampler(stacks[st]["image"],src_meta = stacks[st]["image_meta_dict"], dst_meta = stacks[0]["image_meta_dict"])
        stacks[st]["image_meta_dict"]["filename_or_obj"] = file_obj
    return stacks
 

def save_to_nifti(stacks):
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
        
    

       
filenames = ["10_3T_nody_001.nii.gz",
"10_3T_nody_002.nii.gz",
"14_3T_nody_001_seg.nii.gz",
"14_3T_nody_001.nii.gz",
"14_3T_nody_002.nii.gz",
"21_3T_nody_001.nii.gz",
"21_3T_nody_002.nii.gz",
"23_3T_nody_001.nii.gz",
"23_3T_nody_002.nii.gz",
"mask_10_3T_brain_smooth.nii.gz"]    
    
    
    
folder = "sample_data"
stacks = load_files(folder, filenames)
stacks = resample_to_world_coord(stacks)
save_to_nifti(stacks)

resampler = monai.transforms.ResampleToMatch()
spat_resampler = monai.transforms.SpatialResampled(keys=["image"], meta_keys=["image_meta_dict"], meta_src_keys=["affine"], meta_dst_keys=["dst_affine"])

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




target_dict_2["image"], target_dict_2["image_meta_dict"] = resampler(target_dict_2["image"], src_meta = target_dict_2["image_meta_dict"], dst_meta = target_dict_1["image_meta_dict"])
mask_dict["image"], mask_dict["image_meta_dict"] = resampler(mask_dict["image"], src_meta = mask_dict["image_meta_dict"], dst_meta = target_dict_1["image_meta_dict"])

# target_dict_1 = spacing(target_dict_1)
# target_dict_2 = spacing(target_dict_2)
# mask_dict = spacing(mask_dict)

# target_dict_1 = spat_resampler(target_dict_1)
# target_dict_2 = spat_resampler(target_dict_2)
# mask_dict = spat_resampler(mask_dict)




# grid = monai_utils.create_grid([200,100,100],backend = "torch")
# ext_1 = monai_utils.get_extreme_points(target_dict_1["image"])


# target_1_mask = {}
# target_1_mask["image"], target_1_mask["image_meta_dict"] = resampler(mask, src_meta = mask_dict["image_meta_dict"], dst_meta = target_dict_1["image_meta_dict"])



# masked_indices_1 = t.nonzero(target_1_mask["image"])
# min_indices_1 = t.tensor([t.min(masked_indices_1[:,1]), t.min(masked_indices_1[:,2]),t.min(masked_indices_1[:,3])])
# max_indices_1 = t.tensor([t.max(masked_indices_1[:,1]), t.max(masked_indices_1[:,2]),t.max(masked_indices_1[:,3])])

# roi_size_1 = max_indices_1 - min_indices_1
# roi_center_1 = min_indices_1 + t.ceil((roi_size_1)/2)
# cropper_1 = monai.transforms.SpatialCrop(roi_center = roi_center_1, roi_size = roi_size_1)


# target_2_mask = {}
# target_2_mask["image"], target_2_mask["image_meta_dict"] = resampler(mask, src_meta = mask_dict["image_meta_dict"], dst_meta = target_dict_2["image_meta_dict"])

# masked_indices_2 = t.nonzero(target_2_mask["image"])
# min_indices_2 = t.tensor([t.min(masked_indices_2[:,1]), t.min(masked_indices_2[:,2]),t.min(masked_indices_2[:,3])])
# max_indices_2 = t.tensor([t.max(masked_indices_2[:,1]), t.max(masked_indices_2[:,2]),t.max(masked_indices_2[:,3])])

# roi_size_2 = max_indices_2 - min_indices_2
# roi_center_2 = min_indices_2 + t.ceil((roi_size_2)/2)
# cropper_2 = monai.transforms.SpatialCrop(roi_center = roi_center_2, roi_size = roi_size_2)



# target_dict_1["image"] = cropper_1(target_dict_1["image"])
# target_dict_2["image"] = cropper_2(target_dict_2["image"])

# target_dict_2["image"], target_dict_2["image_meta_dict"] = resampler(target_dict_2["image"], src_meta = target_dict_2["image_meta_dict"], dst_meta = target_dict_1["image_meta_dict"])
# # target_dict_1["image"], target_dict_1["image_meta_dict"] = resampler(target_dict_1["image"], src_meta=target_dict_1["image_meta_dict"], dst_meta=target_1_mask["image_meta_dict"])

folder = "test_reconstruction_monai"
folder2 = "trans"
path = os.path.join(folder,folder2)
nifti_saver = monai.data.NiftiSaver(output_dir=path, 
                                    resample = False, mode = mode, padding_mode = "zeros",
                                    separate_folder=False)


target_dict_1["image_meta_dict"].update({"filename_or_obj":  "target_1"})
target_dict_2["image_meta_dict"].update({"filename_or_obj":  "target_2"})
mask_dict["image_meta_dict"].update({"filename_or_obj":  "mask"})

nifti_saver.save(target_dict_1["image"], meta_data=target_dict_1["image_meta_dict"])
nifti_saver.save(target_dict_2["image"], meta_data=target_dict_2["image_meta_dict"])
nifti_saver.save(mask_dict["image"], meta_data=mask_dict["image_meta_dict"])



#make first dimension the slices
#target_dict = orientation(target_dict)