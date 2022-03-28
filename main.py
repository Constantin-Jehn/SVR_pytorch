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
)
import numpy as np
import os
from copy import deepcopy

#create the slices as 3d tensors
def slices_from_volume(volume_dict):
    image = volume_dict["image"]
    im_slices = list()
    for i in range (0,image.shape[1]):
        slice_dict = deepcopy(target_dict)
        tmp = slice_dict["image"]
        tmp[:,:i,:,:] = 0
        tmp[:,i+1:,:,:] = 0
        slice_dict["image"] = tmp
        im_slices.append(slice_dict)
    return im_slices


def create_volume_dict(folder, filename):
    path = os.path.join(folder, filename)
    # load data
    target_dicts = [{"image": path}]
    loader = LoadImaged(keys = ("image"))
    target_dict = loader(target_dicts[0])
    return target_dict

def preprocess(target_dict, pixdim):
    add_channel = AddChanneld(keys=["image"])
    target_dict = add_channel(target_dict)
    #make first dimension the slices
    orientation = monai.transforms.Orientationd(keys = ("image"), axcodes="PLI")
    target_dict = orientation(target_dict)
    
    #resample image to desired pixdim
    mode = "bilinear"
    spacing = Spacingd(keys=["image"], pixdim=pixdim, mode=mode)
    target_dict = spacing(target_dict)
    return target_dict
#reconstruct the 3d image as sum of the slices
def reconstruct_3d_volume(im_slices):
    n_slices = len(im_slices)
    tmp = np.zeros(im_slices[0]["image"].shape)
    for i in range(0,n_slices):
        tmp += im_slices[i]["image"]
    #update target_dict
    target_dict["image"] = tmp
    return target_dict


mode = "bilinear"
folder = 'sample_data'
filename = '10_3T_nody_001.nii.gz'
target_dict = create_volume_dict(folder,filename)

#save the slices for later comparison
ground_truth = target_dict["image"]
ground_pixdim = target_dict["image_meta_dict"]["pixdim"]

pixdim = (3,3,3)
target_dict = preprocess(target_dict,pixdim)

#visualize 
image = target_dict["image"]
plt.figure("visualise", (8, 4))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(image[0, 30, :, :], cmap="gray")
plt.show()

im_slices = slices_from_volume(target_dict)


k = 25
im_slice = im_slices[k]["image"]
plt.figure("data:", (8, 4))
plt.subplot(1, 2, 1)
plt.title("slice")
plt.imshow(im_slice[0,k,:,:], cmap="gray")
plt.subplot(1,2,2)
plt.title("no data")
plt.imshow(im_slice[0,k+1,:,:], cmap="gray")
plt.show()

#how to do rotations

rotations = [0, 0 ,0 ]
translations = [0, 0 , 0]
affine_trans = monai.transforms.Affine(rotations, translations)
im_slices[k]["image"], im_slices[k]["image_meta_dict"]["affine"] = affine_trans(im_slices[k]["image"])


im_slice = im_slices[k]["image"]
plt.figure("data:", (8, 4))
plt.subplot(1, 3, 1)
plt.title("no data")
plt.imshow(im_slice[0,k+1,:,:], cmap="gray")
plt.subplot(1, 3, 2)
plt.title("rotated")
plt.imshow(im_slice[0,k,:,:], cmap="gray")
plt.subplot(1,3,3)
plt.title("no data")
plt.imshow(im_slice[0,k-1,:,:], cmap="gray")
plt.show()


#resample to initial spacing
target_ict = reconstruct_3d_volume(im_slices)
spacing = Spacingd(keys = ["image"], pixdim = ground_pixdim, mode = mode)
target_dict = spacing(target_dict)

#save
folder = "test_reconstruction_monai"
path = os.path.join(folder)
nifti_saver = monai.data.NiftiSaver(output_dir=path, output_postfix=".nii.gz", 
                                    resample = False, mode = mode, padding_mode = "zeros")
nifti_saver.save(target_dict["image"], meta_data=target_dict["image_meta_dict"])




    