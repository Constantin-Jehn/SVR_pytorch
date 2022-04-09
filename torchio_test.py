import torchio as tio
import torch as t
import numpy as np
import os

from transformation import load_files


filenames = ["10_3T_nody_001.nii.gz",
"10_3T_nody_002.nii.gz",
"14_3T_nody_001.nii.gz",
"14_3T_nody_002.nii.gz",
"21_3T_nody_001.nii.gz",
"21_3T_nody_002.nii.gz",
"23_3T_nody_001.nii.gz",
"23_3T_nody_002.nii.gz"]
file_mask = "mask_10_3T_brain_smooth.nii.gz"
file_world = "world.nii.gz"
pixdim = (1.0, 1.0, 1.0)

src_folder = "sample_data"
dst_folder = "cropped_images"
src_folder = "sample_data"
dst_folder = "cropped_images"

stacks, mask = load_files(src_folder, filenames, file_mask)

fetal_brain = stacks[2]


#fetal_brain=tio.ScalarImage('sample_data/14_3T_nody_001.nii.gz')

#mask=tio.LabelMap('sample_data/mask_10_3T_brain_smooth.nii.gz')

resampler = tio.transforms.Resample(os.path.join(src_folder,filenames[2]))

mask =resampler(mask)

subject = tio.Subject(
    fetal_brain = fetal_brain,
    mask = mask)

masked_indices_1 = t.nonzero(subject["mask"]["data"])
min_indices = np.array([t.min(masked_indices_1[:,1]).item(), t.min(masked_indices_1[:,2]).item(),t.min(masked_indices_1[:,3]).item()])
max_indices = np.array([t.max(masked_indices_1[:,1]).item(), t.max(masked_indices_1[:,2]).item(),t.max(masked_indices_1[:,3]).item()])
roi_size = (max_indices - min_indices) 


transform = tio.CropOrPad(
    list(roi_size),
    mask_name= 'mask')

cropped = transform(subject)

cropped.fetal_brain.save('sample_data/14_tio_crop.nii.gz')