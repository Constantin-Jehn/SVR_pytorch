import os
from matplotlib import transforms
import nibabel as nib
import matplotlib.pyplot as plt
from sympy import rotations
import torch as t
import dill
from scipy.spatial.transform import Rotation as R

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
    rotations : t.tensor (3x3)
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
    

        