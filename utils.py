import os
import nibabel as nib
import matplotlib.pyplot as plt
import torch as t
import dill
#import volume
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
import pytorch3d.transforms as transforms

def nii_to_torch(folder, filename):
    """opens nifti file from /data and return data and affine as torch tensors"""
    path = os.path.join(folder, filename)
    img = nib.load(path)
    epi_image = t.tensor(img.get_fdata())
    affine = t.tensor(img.affine)
    zooms = t.tensor(img.header.get_zooms())
    return epi_image, affine, zooms

def torch_to_nii(data,affine):
    data = data.numpy()
    affine = affine.numpy()
    new_img = nib.Nifti1Image(data,affine)
    return new_img

def save_nifti(nifti_image, folder, filename):
    path = os.path.join(folder, filename + '.nii.gz')
    nib.save(nifti_image, path)
    
def save_target(obj, folder, filename):
    path = os.path.join(folder, filename)
    #dest = filename + ".pickle"
    dill.dump(obj, file = open(path + ".pickle", "wb"))

def open_target(folder,filename):
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
    """Vectorized Gaussian point spread function"""
    return t.exp(-t.div(offset[:,0]**2,sigmas[0]**2) -t.div(offset[:,1]**2,sigmas[2]**2) -t.div(offset[:,2]**2,sigmas[2]**2)).float()

def ncc(X,X_prime):
    """
    Function to calculate normalize cross-correlation, variables according to Hill 2000
    input X: reference image, X_prime transformed image
    """
    # corners_X, corners_X_prime = X.corners(), X_prime.corners()
    # #overlap region
    # lower_bound, upper_bound  = t.max(corners_X[:,0], corners_X_prime[:,0]).unsqueeze(1), t.min(corners_X[:,1],corners_X_prime[:,1]).unsqueeze(1)
    # X_0 = t.cat((lower_bound,upper_bound), dim = 1)
    # mask_X = t.logical_and(t.all(X.p_r[:3,:].transpose(0,1) >= X_0[:,0],dim = 1),  t.all(X.p_r[:3,:].transpose(0,1) <= X_0[:,1], dim = 1))
    # mask_X_prime = t.logical_and(t.all(X_prime.p_r[:3,:].transpose(0,1) >= X_0[:,0], dim = 1),  t.all(X_prime.p_r[:3,:].transpose(0,1) <= X_0[:,1], dim = 1))
    #indices_X, indices_X_prime = t.where(mask_X), t.where(mask_X_prime)
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

def create_T(rotations, translations):
    """
    

    Parameters
    ----------
    rotations : t.tensor ( 3x3)
        DESCRIPTION.
    translations : t.tensor (1x3)
        translations

    Returns
    -------
    T : TYPE
        DESCRIPTION.

    """
    rotation = transforms.euler_angles_to_matrix(rotations, "XYZ")
    #rotation  = R.from_euler('xyz',rotations, degrees = True)
    #T = t.eye(4)
    #translation = unsqueeze(1)
    bottom = t.tensor([0,0,0,1])
    trans = t.cat((rotation,translations.unsqueeze(1)),dim=1)
    T = t.cat((trans,bottom.unsqueeze(0)),dim = 0)
    return T
    

        