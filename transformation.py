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
from copy import deepcopy


pixdim = (3,3,3)
mode = "bilinear"
folder = "sample_data"
file1 = "14_3T_nody_001.nii.gz"
file2 = "10_3T_nody_001.nii.gz"
file3 = "mask_10_3T_brain_smooth.nii.gz"

path1 = os.path.join(folder, file1)
path2 = os.path.join(folder,file2)
path3 = os.path.join(folder,file3)


add_channel = AddChanneld(keys=["image"])
orientation = monai.transforms.Orientationd(keys = ("image"), axcodes="PLI")
spacing = Spacingd(keys=["image"], pixdim=pixdim, mode=mode)
to_tensor = ToTensord(keys = ("image"))


target_dicts = [{"image": path1}, {"image": path2}, {"image": path3}]
loader = LoadImaged(keys = ("image"))

target_dict_1 = loader(target_dicts[0])
target_dict_2 = loader(target_dicts[1])
mask_dict = loader(target_dicts[2])


target_dict_1 = to_tensor(target_dict_1)
target_dict_2 = to_tensor(target_dict_2)
mask_dict = to_tensor(mask_dict)

#ground_pixdim = target_dict["image_meta_dict"]["pixdim"]
target_dict_1 = add_channel(target_dict_1)
target_dict_2 = add_channel(target_dict_2)
mask_dict = add_channel(mask_dict)
#get box around brain
mask = mask_dict["image"]


mask_resampler = monai.transforms.ResampleToMatch()

target_1_mask = {}
target_1_mask["image"], target_1_mask["image_meta_dict"] = mask_resampler(mask, src_meta = mask_dict["image_meta_dict"], dst_meta = target_dict_1["image_meta_dict"])

masked_indices = t.nonzero(target_1_mask["image"])
min_indices = t.tensor([t.min(masked_indices[:,1]), t.min(masked_indices[:,2]),t.min(masked_indices[:,3])])
max_indices = t.tensor([t.max(masked_indices[:,1]), t.max(masked_indices[:,2]),t.max(masked_indices[:,3])])


roi_size = max_indices - min_indices
roi_center = min_indices + t.ceil((roi_size)/2)

cropper = monai.transforms.SpatialCrop(roi_center = roi_center, roi_size = roi_size)

target_dict_1["image"] = cropper(target_dict_1["image"])
target_dict_1["image"], target_dict_1["image_meta_dict"] = mask_resampler(target_dict_1["image"], src_meta=target_dict_1["image_meta_dict"], dst_meta=target_1_mask["image_meta_dict"])



folder = "sample_data"
path = os.path.join(folder)
nifti_saver = monai.data.NiftiSaver(output_dir=path, 
                                    resample = False, mode = mode, padding_mode = "zeros",
                                    separate_folder=False)

nifti_saver.save(target_dict_1["image"], meta_data=target_dict_1["image_meta_dict"])








#make first dimension the slices
#target_dict = orientation(target_dict)