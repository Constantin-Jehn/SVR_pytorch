import torch as t
from torchvision import transforms

def PSF_Gauss(offset,sigmas = [10,10,10]):
    """Gaussian pointspreadfunction higher sigma --> sharper kernel"""
    return t.exp(-(offset[0]**2)/sigmas[0]**2 - (offset[1]**2)/sigmas[1]**2 - (offset[2]**2)/sigmas[2]**2).float()
    
def PSF_Gauss_vec(offset, sigmas = [10,10,10]):
    """Vectorized Gaussian point spread function"""
    return t.exp(-t.div(offset[:,0]**2,sigmas[0]**2) -t.div(offset[:,1]**2,sigmas[2]**2) -t.div(offset[:,2]**2,sigmas[2]**2)).float()

"""Coordinate transforms"""
def f(t_image, t_affine):
    """Function to map from Voxel coordinates to reference coordinates """
    M = t_affine[:3,:3]
    abc = t_affine[:3,3]
    epi_center = ((t.tensor(t_image.shape)-1)/2).double()
    return t.linalg.multi_dot([M,epi_center]) + abc     

def f_inv(t_image, t_affine):
    """Function to map from reference coordinates to Voxel """
    inv = t.linalg.inv(t_affine)
    epi_center = t.cat((((t.tensor(t_image.shape)-1)/2).double(),t.tensor([1])))
    voxel = t.linalg.multi_dot(inv,epi_center)
    return voxel.chunk(3)[0]

"""In-plane transformation of slices"""
def R_z(img, index, angle):
    """Function to perform in-plane rotation"""
    img_slice = img[None,:,:,index]
    rotated_slice = transforms.functional.rotate(img_slice, angle)
    img[:,:,index] = t.squeeze(rotated_slice)
    return img

def T_x(img,index,translation):
    """Function to perfron in-plane translation in x-direction"""
    img_slice = img[None,:,:,index]
    translated_slice = transforms.functional.affine(img_slice,0,[0,translation],1,0)
    img[:,:,index] = t.squeeze(translated_slice)
    return img

def T_y(img,index,translation):
    """Function to perfron in-plane translation in x-direction"""
    img_slice = img[None,:,:,index]
    translated_slice = transforms.functional.affine(img_slice,0,[translation,0],1,0)
    img[:,:,index] = t.squeeze(translated_slice)
    return img


