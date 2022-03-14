import os
import nibabel as nib
import matplotlib.pyplot as plt
import torch as t


def nii_to_torch(filename):
    """opens nifti file from /data and return data and affine as torch tensors"""
    path = os.path.join("data", filename)
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

def save_nifti(nifti_image, filename):
    path = os.path.join('data', filename + '.nii.gz')
    nib.save(nifti_image, path)

def show_stack(t_image):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, t_image.shape[2])
    for i in range(t_image.shape[2]):
      axes[i].imshow(t.transpose(t_image[:,:,i],1,0), cmap="gray", origin="lower")
    return 0
      