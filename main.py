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
import numpy as np
import torch as t
from copy import deepcopy
from SVR_optimizer import SVR_optimizer

def optimize():
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    
    # filenames = ["10_3T_nody_001.nii.gz",
    #                 "14_3T_nody_001.nii.gz"]

    filenames = ["10_3T_nody_001.nii.gz",
                  
                  "14_3T_nody_001.nii.gz",
                 
                  "21_3T_nody_001.nii.gz",
                  
                  "23_3T_nody_001.nii.gz"]
    
    file_mask = "mask_10_3T_brain_smooth.nii.gz"
    
    pixdims = [(4.0,4.0,4.0),(2.5,2.5,2.5),(1.5,1.5,1.5),(1.0,1.0,1.0)]

    src_folder = "sample_data"
    prep_folder = "cropped_images"
    src_folder = "sample_data"
    mode = "bilinear"
    
    
    svr_optimizer = SVR_optimizer(src_folder, prep_folder, filenames, file_mask,pixdims, device, mode = mode)
    
    epochs = 1
    inner_epochs = 2
    lr = 0.0005
    loss_fnc = "ncc"
    opt_alg = "Adam"
    
    #world_stack, loss_log = svr_optimizer.optimize_multiple_stacks(epochs, inner_epochs, lr, loss_fnc=loss_fnc, opt_alg=opt_alg)
    
    fixed_images = svr_optimizer.fixed_images
    #fixed_image["image"] = t.squeeze(fixed_image["image"]).unsqueeze(0)
    
    folder = "test_reconstruction_monai"
    folder2 = "fixed_images"
    path = os.path.join(folder,folder2)
    
    
    nifti_saver = monai.data.NiftiSaver(output_dir=path, 
                                        resample = False, mode = mode, padding_mode = "zeros",
                                        separate_folder=False)
    #save_to = 'reconstruction_' + opt_alg + '_(' + str(pixdim[0]).replace('.',',')  +'-'+ str(pixdim[1]).replace('.',',') +'-'+ str(pixdim[2]).replace('.',',') + ')_lr' + str(lr).replace('.',',') + '_' + str(epochs) + '_' + mode
    
    #world_stack["image_meta_dict"]["filename_or_obj"] = save_to + "nii.gz"
    
    for i in range(0,len(fixed_images)):
        fixed_images[i]["image"] = t.squeeze(fixed_images[i]["image"]).unsqueeze(0)
        fixed_images[i]["image_meta_dict"]["filename_or_obj"] = "fixed_image_" +str(i) +  ".nii.gz"
        nifti_saver.save(fixed_images[i]["image"], meta_data=fixed_images[i]["image_meta_dict"])
    
    #nifti_saver.save(world_stack["image"], meta_data=world_stack["image_meta_dict"])
    

if __name__ == '__main__':
    
    optimize()
