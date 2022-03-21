import os
import nibabel as nib
import matplotlib.pyplot as plt
import torch as t
import dill
import volume


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


# def ncc(X:volume.volume,X_prime:volume.volume):
#     """
#     Function to calculate normalize cross-correlation, variables according to Hill 2000
#     input X: reference image, X_prime transformed image
#     """
#     corners_X, corners_X_prime = X.corners(), X_prime.corner()
#     #overlap region
#     X_0 = t.cat(t.max(corners_X[:,0], corners_X_prime[:,0]), t.min(corners_X[:,1],corners_X_prime[:,1]), dim = 1)
#     indices_X, indices_X_prime = t.where(X.p_r[:3])
    
      