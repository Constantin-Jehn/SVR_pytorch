import torchio as tio
import torch as t
import numpy as np



fetal_brain=tio.ScalarImage('sample_data/14_3T_nody_001.nii.gz')

mask=tio.LabelMap('sample_data/mask_10_3T_brain_smooth.nii.gz')

resampler = tio.transforms.Resample(fetal_brain)

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