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
import datetime
import json
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["taskset"] = "21-40"

    
def optimize():
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    """
    filenames = ["10_3T_nody_001.nii.gz",
                
                "14_3T_nody_001.nii.gz"]

    """
    filenames = ["14_3T_nody_001.nii.gz",
                "14_3T_nody_002.nii.gz",

                "10_3T_nody_001.nii.gz",
                "10_3T_nody_002.nii.gz",
                
                "21_3T_nody_001.nii.gz",
                "21_3T_nody_002.nii.gz",
                
                "23_3T_nody_001.nii.gz",
                "23_3T_nody_002.nii.gz"]
   
    file_mask = "mask_10_3T_brain_smooth.nii.gz"
   
    pixdims_float = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]

    pixdims = [(x,x,x) for x in pixdims_float]

    mode = "bicubic"
    tio_mode = "welch"
    
    epochs = 6
    inner_epochs = 2
    
    loss_fnc = "ncc"
    opt_alg = "Adam"


    src_folder = "sample_data"
    prep_folder = "cropped_images"
    src_folder = "sample_data"

    current_date = datetime.datetime.now()
    result_string =   "rec_" + str(current_date.month) + "_" + str(current_date.day) + "_" +   str(current_date.hour) + "_"  + str(current_date.minute) + "_Ep_" + str(epochs)
    result_folder = os.path.join("results", result_string)
    tensor_board_folder = os.path.join("runs", result_string)
    
    try:
        os.mkdir(result_folder)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    #prev 0.0015
    lr = 0.0015
    #lambda function for setting learning rate
    lambda1 = lambda epoch: [0.1,0.3,0.5,0.8,1,1][epoch] if epoch  < 5  else 1
    #lambda1 = lambda epoch: 1 if epoch in [0] else 0.5 if epoch in [1] else 0.25 if epoch in [2,3,4] else 0.2
    #lambda1 = lambda epoch: 1 if epoch in [0] else 0.2

    sav_gol_kernel_size = 13
    sav_gol_order = 4
    psf_string = "Sav_Gol"

    if psf_string == "Sav_Gol":
        PSF = monai.networks.layers.SavitzkyGolayFilter(sav_gol_kernel_size,sav_gol_order,axis=3,mode="zeros")
        PSF_doc = {"name": psf_string, "kernel_size": sav_gol_kernel_size, "order": sav_gol_order}
    elif psf_string == "Gaussian":
        sigma_gauss = [1.2 * (1/2.35), 1.2 * (1/2.35), (1/2.35)]
        PSF= monai.networks.layers.GaussianFilter(spatial_dims = 3, sigma = sigma_gauss)
        PSF_doc = {"name": psf_string, "sigma": sigma_gauss}
    else:
        assert('Choose Sav_Gol or Gaussian as PSF')

    loss_kernel_size = 31

    from_checkpoint = False
    last_rec_file = "reconstruction_volume_10.nii.gz"
    last_epoch = 10
    roi_only = True

    parameter_file = {
        "Result_folder": result_folder,
        "Interpolation": {
            "monai_interpolation_mode": mode,
            "tochio_interpolation_mode": tio_mode
            },
        "Opimization":{
            "Epochs": epochs,
            "Inner_epochs": inner_epochs,
            "Loss_fnc": loss_fnc,
            "Loss_kernel_size": loss_kernel_size,
            "Optimisation_algorithm": opt_alg,
            "Learning_rate": lr,
            },
        "PSF": PSF_doc
    }
    parameter_file_dest = os.path.join(result_folder,"parameters.json")
    out_file = open(parameter_file_dest, "w")
    json.dump(parameter_file,out_file, indent=6)
    out_file.close()

    svr_optimizer = SVR_optimizer(src_folder, prep_folder, result_folder, filenames, file_mask,pixdims, device, PSF, loss_kernel_size, monai_mode = mode, tio_mode = tio_mode, roi_only=roi_only)
    svr_optimizer.optimize_volume_to_slice(epochs, inner_epochs, lr, PSF, lambda1, loss_fnc=loss_fnc, opt_alg=opt_alg, tensorboard=True, tensorboard_path=tensor_board_folder,from_checkpoint=from_checkpoint, last_rec_file=last_rec_file, last_epoch = last_epoch)
    
if __name__ == '__main__':
    optimize()
    #preprocess()
