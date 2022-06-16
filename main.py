import matplotlib.pyplot as plt
import monai
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
    
    pixdims = [(1.0,1.0,1.0),(2.0,2.0,2.0),(1.5,1.5,1.5),(1.1,1.1,1.1)]

    src_folder = "sample_data"
    prep_folder = "cropped_images"
    src_folder = "sample_data"
    result_folder = os.path.join("results","preprocessing_gaussian_0_5")
    mode = "bicubic"
    tio_mode = "welch"
    
    svr_preprocessor = Preprocesser(src_folder, prep_folder, result_folder, filenames, file_mask, device, mode, tio_mode)
    fixed_images, stacks, slice_dimensions = svr_preprocessor.preprocess_stacks_and_common_vol(init_pix_dim = pixdims[0], save_intermediates=True)
    svr_preprocessor.save_stacks(stacks,'out')
    
    fixed_images["image"] = t.squeeze(fixed_images["image"]).unsqueeze(0)
    
    folder = "preprocessing_gaussian"
    path = os.path.join(folder)
    nifti_saver = monai.data.NiftiSaver(output_dir=path, 
                                        resample = False, mode = mode, padding_mode = "zeros",
                                        separate_folder=False)
    
    fixed_images["image_meta_dict"]["filename_or_obj"] = "pre_registered"
    nifti_saver.save(fixed_images["image"], meta_data=fixed_images["image_meta_dict"])
    

def optimize():
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    """
    filenames = ["10_3T_nody_001.nii.gz",
                
                "14_3T_nody_001.nii.gz"]

    """
    filenames = ["10_3T_nody_001.nii.gz",
                
                "14_3T_nody_001.nii.gz",
                
                "21_3T_nody_001.nii.gz",
                
                "23_3T_nody_001.nii.gz"]
    
    file_mask = "mask_10_3T_brain_smooth.nii.gz"
   
    pixdims = [(1.0, 1.0, 1.0),(1.0,1.0,1.0),(1.0,1.0,1.0),(1.0,1.0,1.0),(1.0, 1.0, 1.0),(1.0,1.0,1.0),(1.0,1.0,1.0),(1.0,1.0,1.0),(1.0, 1.0, 1.0),(1.0,1.0,1.0),(1.0,1.0,1.0),(1.0,1.0,1.0),(1.0, 1.0, 1.0),(1.0,1.0,1.0),(1.0,1.0,1.0),(1.0,1.0,1.0),(1.0,1.0,1.0),(1.0, 1.0, 1.0),(1.0,1.0,1.0),(1.0,1.0,1.0),(1.0,1.0,1.0)]

    src_folder = "sample_data"
    prep_folder = "cropped_images"
    src_folder = "sample_data"
    result_string = "Ep_20_voxelmorph_ncc_16_06_16_00"
    result_folder = os.path.join("results", result_string)
    tensor_board_folder = os.path.join("runs", result_string)
    
    try:
        os.mkdir(result_folder)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    mode = "bicubic"
    tio_mode = "welch"
    
    epochs = 20
    inner_epochs = 2
    lr = 0.01
    loss_fnc = "ncc"
    opt_alg = "Adam"
    sav_gol_kernel_size = 13
    sav_gol_order = 4

    #lambda function for setting learning rate
    lambda1 = lambda epoch: [0.5,0.25,0.2,0.2,0.2][epoch] if epoch  < 5  else 0.1 if epoch < 10 else 0.05
    #lambda1 = lambda epoch: 1 if epoch in [0] else 0.5 if epoch in [1] else 0.25 if epoch in [2,3,4] else 0.2
    #lambda1 = lambda epoch: 1 if epoch in [0] else 0.2

    PSF = monai.networks.layers.SavitzkyGolayFilter(sav_gol_kernel_size,sav_gol_order,axis=3,mode="zeros")
    #PSF_alternative = monai.transforms.GaussianSmooth(sigma = [0.1,0.1,0.5])

    from_checkpoint = False
    last_rec_file = "reconstruction_volume_10.nii.gz"
    last_epoch = 10
    roi_only = True

    svr_optimizer = SVR_optimizer(src_folder, prep_folder, result_folder, filenames, file_mask,pixdims, device, PSF, monai_mode = mode, tio_mode = tio_mode, roi_only=roi_only)
    svr_optimizer.optimize_volume_to_slice(epochs, inner_epochs, lr, PSF, lambda1, loss_fnc=loss_fnc, opt_alg=opt_alg, tensorboard=True, tensorboard_path=tensor_board_folder,from_checkpoint=from_checkpoint, last_rec_file=last_rec_file, last_epoch = last_epoch)
    
if __name__ == '__main__':
    optimize()
    #preprocess()
