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
from SVR_Preprocessor import Preprocesser
import errno

def preprocess():
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    
    filenames = ["10_3T_nody_001.nii.gz",
                  
                  "14_3T_nody_001.nii.gz",
                 
                  "21_3T_nody_001.nii.gz",
                  
                  "23_3T_nody_001.nii.gz"]
    
    file_mask = "mask_10_3T_brain_smooth.nii.gz"
    
    pixdims = [(2.5,2.5,2.5),(2.0,2.0,2.0),(1.5,1.5,1.5),(1.1,1.1,1.1)]

    src_folder = "sample_data"
    prep_folder = "cropped_images"
    src_folder = "sample_data"
    mode = "bicubic"
    
    svr_preprocessor = Preprocesser(src_folder, prep_folder, filenames, file_mask,pixdims, device, mode = mode)
    fixed_images, stacks = svr_preprocessor.preprocess_stacks_and_common_vol(save_intermediates=True)
    svr_preprocessor.save_stacks(stacks,'out')
    
    fixed_images["image"] = t.squeeze(fixed_images["image"]).unsqueeze(0)
    
    folder = "preprocessing"
    path = os.path.join(folder)
    nifti_saver = monai.data.NiftiSaver(output_dir=path, 
                                        resample = False, mode = mode, padding_mode = "zeros",
                                        separate_folder=False)
    
    fixed_images["image_meta_dict"]["filename_or_obj"] = "pre_registered"
    nifti_saver.save(fixed_images["image"], meta_data=fixed_images["image_meta_dict"])
    

def optimize():
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    
    filenames = ["10_3T_nody_001.nii.gz",
                
                "14_3T_nody_001.nii.gz"]

    """"
    filenames = ["10_3T_nody_001.nii.gz",
                
                "14_3T_nody_001.nii.gz",
                
                "21_3T_nody_001.nii.gz",
                
                "23_3T_nody_001.nii.gz"]
    """
    file_mask = "mask_10_3T_brain_smooth.nii.gz"
    
    pixdims = [(2.0, 2.0, 2.0),(1.5,1.5,1.5),(1.0,1.0,1.0),(1.0,1.0,1.0)]

    src_folder = "sample_data"
    prep_folder = "cropped_images"
    src_folder = "sample_data"
    result_folder = os.path.join("results","two_stacks_multi_res")
    
    try:
        os.mkdir(result_folder)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    mode = "nearest"
    
    
    svr_optimizer = SVR_optimizer(src_folder, prep_folder, result_folder, filenames, file_mask,pixdims, device, mode = mode)
    
    epochs = 2
    inner_epochs = 1
    lr = 0.001
    loss_fnc = "mi"
    opt_alg = "Adam"
    
    svr_optimizer.optimize_volume_to_slice(epochs, inner_epochs, lr, loss_fnc=loss_fnc, opt_alg=opt_alg)
    
if __name__ == '__main__':
    
    optimize()
    #preprocess()
