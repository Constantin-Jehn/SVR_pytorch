import matplotlib.pyplot as plt
import monai
import os
import numpy as np
import torch as t
from copy import deepcopy
from SVR_optimizer import SVR_optimizer
from SVR_Preprocessor import Preprocesser
import errno
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["taskset"] = "21-40"

    
def optimize():
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    """
    filenames = ["10_3T_nody_001.nii.gz",
                
                "14_3T_nody_001.nii.gz"]

    """
    filenames = ["14_3T_nody_001.nii.gz",

                "10_3T_nody_001.nii.gz",
                
                "21_3T_nody_001.nii.gz",
                
                "23_3T_nody_001.nii.gz"]
   
    file_mask = "mask_10_3T_brain_smooth.nii.gz"
   
    pixdims_float = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    pixdims = [(x,x,x) for x in pixdims_float]

    src_folder = "sample_data"
    prep_folder = "cropped_images"
    src_folder = "sample_data"
    result_string = "Ep_5_prereg_27_06_16:15"
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
    
    epochs = 5
    inner_epochs = 2
    lr = 0.0003
    loss_fnc = "ncc"
    opt_alg = "Adam"
    sav_gol_kernel_size = 13
    sav_gol_order = 4

    #lambda function for setting learning rate
    lambda1 = lambda epoch: [1,1,1,1,1][epoch] if epoch  < 5  else 0.2 if epoch < 10 else 0.125
    #lambda1 = lambda epoch: 1 if epoch in [0] else 0.5 if epoch in [1] else 0.25 if epoch in [2,3,4] else 0.2
    #lambda1 = lambda epoch: 1 if epoch in [0] else 0.2

    PSF = monai.networks.layers.SavitzkyGolayFilter(sav_gol_kernel_size,sav_gol_order,axis=3,mode="zeros")
    #PSF_alternative = monai.transforms.GaussianSmooth(sigma = [0.1,0.1,0.5])

    loss_kernel_size = 31

    from_checkpoint = False
    last_rec_file = "reconstruction_volume_10.nii.gz"
    last_epoch = 10
    roi_only = False

    svr_optimizer = SVR_optimizer(src_folder, prep_folder, result_folder, filenames, file_mask,pixdims, device, PSF, loss_kernel_size, monai_mode = mode, tio_mode = tio_mode, roi_only=roi_only)
    svr_optimizer.optimize_volume_to_slice(epochs, inner_epochs, lr, PSF, lambda1, loss_fnc=loss_fnc, opt_alg=opt_alg, tensorboard=True, tensorboard_path=tensor_board_folder,from_checkpoint=from_checkpoint, last_rec_file=last_rec_file, last_epoch = last_epoch)
    
if __name__ == '__main__':
    optimize()
    #preprocess()
