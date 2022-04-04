import os
from matplotlib import transforms
import nibabel as nib
import matplotlib.pyplot as plt
from sympy import rotations
import torch as t
import dill
from scipy.spatial.transform import Rotation as R

import monai
from monai.transforms import (
    AddChanneld,
    LoadImage,
    LoadImaged,
    Orientationd,
    Rand3DElasticd,
    RandAffined,
    Spacingd,
    ToTensord
)
from copy import deepcopy

import numpy as np

def nii_to_torch(folder, filename):
    """
    opens nifti file from /data and return data and affine as torch tensors
    """
    path = os.path.join(folder, filename)
    img = nib.load(path)
    epi_image = t.tensor(img.get_fdata())
    affine = t.tensor(img.affine)
    zooms = t.tensor(img.header.get_zooms())
    return epi_image, affine, zooms

def torch_to_nii(data,affine):
    """
    Parameters
    ----------
    data : t.tensor
        tensor of the 3D volume.
    affine : t.tensor (4x4)
        corresponding affine

    Returns
    -------
    new_img : nifti_img

    """
    data = data.numpy()
    affine = affine.numpy()
    new_img = nib.Nifti1Image(data,affine)
    return new_img

def save_nifti(nifti_image, folder, filename):
    """
    Parameters
    ----------
    nifti_image : 
        image to save
    folder : string
        
    filename : string

    Returns
    -------
    None.

    """
    path = os.path.join(folder, filename + '.nii.gz')
    nib.save(nifti_image, path)
    
def save_target(obj, folder, filename):
    """
    Parameters
    ----------
    obj : volume object
        created volume object you want to save
    folder : string
    filename : string
    Returns
    -------
    None.

    """
    path = os.path.join(folder, filename)
    #dest = filename + ".pickle"
    dill.dump(obj, file = open(path + ".pickle", "wb"))

def open_target(folder,filename):
    """
    Parameters
    ----------
    folder : string
    filename : string

    Returns
    -------
    target as volume object

    """
    path = os.path.join(folder, filename)
    return dill.load((open(path + ".pickle","rb")))

def show_stack(t_image):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, t_image.shape[2])
    for i in range(t_image.shape[2]):
      axes[i].imshow(t.transpose(t_image[:,:,i],1,0), cmap="gray", origin="lower")
      
      
def PSF_Gauss(offset,sigmas = [10,10,10]):
    """Gaussian pointspreadfunction higher sigma --> sharper kernel"""
    return t.exp(-(offset[0]**2)/sigmas[0]**2 - (offset[1]**2)/sigmas[1]**2 - (offset[2]**2)/sigmas[2]**2).float()
    
def PSF_Gauss_vec(offset, sigmas = [10,10,10]):
    """
    Gaussian PSF
    Parameters
    ----------
    offset : t.tensor
        difference tensor between voxel centers and transformed pixels in world coordinates
    sigmas : TYPE, optional
        DESCRIPTION. The default is [10,10,10].

    Returns
    -------
    t.tensor
        approximated intensities as voxels
    """
    return t.exp(-t.div(offset[:,0]**2,sigmas[0]**2) -t.div(offset[:,1]**2,sigmas[2]**2) -t.div(offset[:,2]**2,sigmas[2]**2)).float()

def ncc(X,X_prime):
    """
    Function to calculate normalize cross-correlation, variables according to Hill 2000
    input X: reference image, X_prime transformed image
    """
    f_x0, g_x0 = X.p_r[4,:].unsqueeze(0), X_prime.p_r[4,:].unsqueeze(0)
    corr_coef = t.corrcoef(t.cat((f_x0,g_x0), dim = 0))
    ncc = corr_coef[0,1] 
    return ncc
    
def ncc_within_volume(target_volume):
    k = target_volume.X.shape[2]
    ncc = 0
    for sl in range(0,k-1):
        f_x, g_x = target_volume.X[:,:,sl].flatten().unsqueeze(0), target_volume.X[:,:,sl+1].flatten().unsqueeze(0)
        corr_coef = t.corrcoef(t.cat((f_x,g_x),dim = 0))
        #make negative for loss
        ncc -= corr_coef[0,1]
    return ncc

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

def rotation_matrix(angles):
    s = t.sin(angles)
    c = t.cos(angles)
    rot_x = t.cat((t.tensor([1,0,0]),
                  t.tensor([0,c[0],-s[0]]),
                  t.tensor([0,s[0],c[0]])), dim = 0).reshape(3,3)
    
    rot_y = t.cat((t.tensor([c[1],0,s[1]]),
                  t.tensor([0,1,0]),
                  t.tensor([-s[1],0,c[1]])),dim = 0).reshape(3,3)
    
    rot_z = t.cat((t.tensor([c[2],-s[2],0]),
                  t.tensor([s[2],c[2],0]),
                  t.tensor([0,0,1])), dim = 0).reshape(3,3)
    return t.matmul(t.matmul(rot_z, rot_y),rot_x)
    

def create_T(rotations, translations, device):
    """
    Parameters
    ----------
    rotations : t.tensor (1x3)
        convention XYZ
    translations : t.tensor (1x3)
        translations

    Returns
    -------
    T : TYPE
        DESCRIPTION.

    """
    rotation = rotation_matrix(rotations).to(device)
    bottom = t.tensor([0,0,0,1]).to(device)
    trans = t.cat((rotation,translations.unsqueeze(1)),dim=1)
    T = t.cat((trans,bottom.unsqueeze(0)),dim = 0)
    return T
    


#create the slices as 3d tensors
def slices_from_volume(volume_dict):
    """
    Parameters
    ----------
    volume_dict : dictionary
        loaded nifti file as dictionary

    Returns
    -------
    im_slices : list of dictionaries
        each entry in list is a slice of the volume still represented in 3d 
        all entries except for the slice are set to zero

    """
    image = volume_dict["image"]
    im_slices = list()
    for i in range (0,image.shape[2]):
        slice_dict = deepcopy(volume_dict)
        tmp = slice_dict["image"]
        tmp[:,:,:i,:,:] = 0
        tmp[:,:,i+1:,:,:] = 0
        slice_dict["image"] = tmp
        im_slices.append(slice_dict)
    return im_slices


def create_volume_dict(folder, filename, pixdim):
    """
    Parameters
    ----------
    folder : string
        
    filename : string
        
    pixdim : list
        dimensions of voxels

    Returns
    -------
    target_dict : dictionary
        volume representation of nifti
    ground_image : TYPE
        DESCRIPTION.
    ground_pixdim : TYPE
        DESCRIPTION.

    """
    path = os.path.join(folder, filename)
    # load data
    target_dicts = [{"image": path}]
    loader = LoadImaged(keys = ("image"))
    to_tensor = ToTensord(keys = ("image"))
    target_dict = loader(target_dicts[0])
    target_dict = to_tensor(target_dict)
    add_channel = AddChanneld(keys=["image"])
    target_dict = add_channel(target_dict)
    #make first dimension the slices
    orientation = monai.transforms.Orientationd(keys = ("image"), axcodes="PLI")
    target_dict = orientation(target_dict)
    
    ground_image = deepcopy(target_dict["image"])
    ground_pixdim = deepcopy(target_dict["image_meta_dict"]["pixdim"])
    #resample image to desired pixdim
    mode = "bilinear"
    spacing = Spacingd(keys=["image"], pixdim=pixdim, mode=mode)
    target_dict = spacing(target_dict)
    return target_dict, ground_image, ground_pixdim

#reconstruct the 3d image as sum of the slices
def reconstruct_3d_volume(im_slices, target_dict):
    """
    

    Parameters
    ----------
    im_slices : list of dictionaries
        list of (transformed image slices)
    target_dict : dictionary
        volume dictionary to be updated

    Returns
    -------
    target_dict : dictionary
        volume dictionary from current rotated slices

    """
    n_slices = len(im_slices)
    tmp = t.zeros(im_slices[0]["image"].shape)
    for i in range(0,n_slices):
        tmp = tmp + im_slices[i]["image"]
    #update target_dict
    target_dict["image"] = tmp
    return target_dict


def monai_demo():
    mode = "bilinear"
    folder = 'sample_data'
    filename = '10_3T_nody_001.nii.gz'
    path = os.path.join(folder, filename)
    pixdim = (3,3,3)
    
    target_dicts = [{"image": path}]
    loader = LoadImaged(keys = ("image"))
    target_dict = loader(target_dicts[0])
    
    to_tensor = ToTensord(keys = ("image"))
    target_dict = to_tensor(target_dict)
    #ground_pixdim = target_dict["image_meta_dict"]["pixdim"]
    
    add_channel = AddChanneld(keys=["image"])
    target_dict = add_channel(target_dict)
    #adds second "channel for batch"
    #target_dict = add_channel(target_dict)
    #make first dimension the slices
    orientation = monai.transforms.Orientationd(keys = ("image"), axcodes="PLI")
    target_dict = orientation(target_dict)
    
    ground_image, ground_meta = deepcopy(target_dict["image"]), deepcopy(target_dict["image_meta_dict"])
    ground_meta["spatial_shape"] = list(target_dict["image"].shape)[1:]
    
    #resample image to desired pixdim
    mode = "bilinear"
    spacing = Spacingd(keys=["image"], pixdim=pixdim, mode=mode)
    target_dict = spacing(target_dict)
    
    #target_dict = preprocess(target_dict,pixdim)
    target_dict = add_channel(target_dict)
    im_slices = slices_from_volume(target_dict)
    k = 10
    
    im_slice = im_slices[k]["image"]
    plt.figure("data",(8, 4))
    plt.subplot(1, 3, 1)
    plt.title("next slice")
    plt.imshow(im_slice[0,0,k+1,:,:], cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("initial")
    plt.imshow(im_slice[0,0,k,:,:], cmap="gray")
    plt.subplot(1,3,3)
    plt.title("previous slice")
    plt.imshow(im_slice[0,0,k-1,:,:], cmap="gray")
    plt.show()
    
    
    rotations = t.tensor([np.pi/16,0,0])
    translations = t.tensor([0,0,0])
    affine_layer = monai.networks.layers.AffineTransform(mode = "bilinear", normalized = True, padding_mode = "border", align_corners=True)
    affine = create_T(rotations, translations, device="cpu")
    
    trans_im_slices = deepcopy(im_slices)
    im_slices[k]["image"] = affine_layer(im_slices[k]["image"],affine)

    
    im_slice = im_slices[k]["image"]
    plt.figure("data",(8, 4))
    plt.subplot(1, 3, 1)
    plt.title("next slice")
    plt.imshow(im_slice[0,0,k+1,:,:], cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("rotated by layer ")
    plt.imshow(im_slice[0,0,k,:,:], cmap="gray")
    plt.subplot(1,3,3)
    plt.title("previous slice")
    plt.imshow(im_slice[0,0,k-1,:,:], cmap="gray")
    plt.show()
    
    
    affine_trans = monai.transforms.Affine(affine = affine)
    #affine_trans = monai.transforms.Affine(rotate_params=rotations.tolist(), translate_params= translations.tolist())
    trans_im_slices[k]["image"] = t.squeeze(trans_im_slices[k]["image"])
    trans_im_slices[k] = add_channel(trans_im_slices[k])
    trans_im_slices[k]["image"], _ = affine_trans(trans_im_slices[k]["image"])
    
    trans_im_slice = trans_im_slices[k]["image"]
    plt.figure("data",(8, 4))
    plt.subplot(1, 3, 1)
    plt.title("next slice")
    plt.imshow(trans_im_slice[0,k+1,:,:], cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("rotated by transform")
    plt.imshow(trans_im_slice[0,k,:,:], cmap="gray")
    plt.subplot(1,3,3)
    plt.title("previous slice")
    plt.imshow(trans_im_slice[0,k-1,:,:], cmap="gray")
    plt.show()
    
    
    # spatial_size = (84,288,288)
    # src_affine = target_dict["image_meta_dict"]["affine"]
    # img = target_dict["image"]
    # resample_to_match = monai.transforms.ResampleToMatch(padding_mode="zeros")
    # resampled_image, resampled_meta = resample_to_match(img,src_meta = target_dict["image_meta_dict"], dst_meta = ground_meta)
    
    # k = 12
    # plt.figure("data",(8, 4))
    # plt.subplot(1, 2, 1)
    # plt.title("original_image")
    # plt.imshow(ground_image[0,k,:,:], cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.title("resampled")
    # plt.imshow(resampled_image[0,k,:,:], cmap="gray")
    # plt.show()
    # #save
    # folder = "test_reconstruction_monai"
    # path = os.path.join(folder)
    
    # nifti_saver = monai.data.NiftiSaver(output_dir=path, output_postfix=".nii.gz", 
    #                                     resample = False, mode = mode, padding_mode = "zeros")
    # nifti_saver.save(target_dict["image"], meta_data=target_dict["image_meta_dict"])
        