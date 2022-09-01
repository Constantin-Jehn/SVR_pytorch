from ntpath import join
from sys import path
from typing import Tuple
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
import torchio as tio
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["taskset"] = "21-40"


def get_list_of_all_dirs(data_dir:str)->list:
    """_summary_

    Args:
        data_dir (str): directory in project folder containing the folder of good reconstructions

    Returns:
        list: list of all reconstruction folders
    """
    current_dir = os.getcwd()
    data_dir_glob = os.path.join(current_dir,data_dir)
    dir_list = [dir for dir in os.listdir(data_dir_glob) if os.path.isdir(os.path.join(data_dir,dir))]
    dir_list = sorted(dir_list)
    return dir_list

def get_list_of_stacks(data_dir:str, sub_dir:str)->list:
    """gives list of stacks in data_dir/sub_dir, looks for files starting with "stack"

    Args:
        data_dir (str): directory in project folder containing the folder of good reconstructions
        sub_dir (str): sub directory of current stack 

    Returns:
        list: file names of stacks
    """
    current_dir = os.getcwd()
    sub_dir_glob = os.path.join(current_dir,data_dir,sub_dir)
    stack_list = []
    for file in os.listdir(sub_dir_glob):
        if file.startswith("stack"):
            stack_list.append(file)
    return stack_list

def get_SVR_reconstruction(data_dir:str, sub_dir:str)->tio.ScalarImage:
    """returns SVR_reconstruction as tio Scalar image

    Args:
        data_dir (str): directory in project folder containing the folder of good reconstructions
        sub_dir (str): sub directory of current stack 

    Returns:
        tio.ScalarImage: the output of the SVR reconstruction to match image size and cropping
    """
    current_dir = os.getcwd()
    svr_output_dir_glob = os.path.join(current_dir,data_dir,sub_dir,"outputSVR.nii.gz")
    tio_svr_output = tio.ScalarImage(svr_output_dir_glob)
    return tio_svr_output

def get_preregistration(data_dir:str, sub_dir:str)->Tuple:
    """return preregistration image as tio Scalar image

    Args:
        data_dir (str): directory in project folder containing the folder of good reconstructions
        sub_dir (str): sub directory of current stack 


    Returns:
        tio.ScalarImage: tio Scalar image of preregstired file
    """
    current_dir = os.getcwd()
    sub_dir_glob = os.path.join(current_dir,data_dir,sub_dir)
    for file in os.listdir(sub_dir_glob):
        if file.endswith("-1.nii.gz"):
            pre_reg_tio = tio.ScalarImage(os.path.join(sub_dir_glob,file))
            prereg_path = os.path.join(sub_dir_glob,file)
    return pre_reg_tio, prereg_path

def get_first_iteration(data_dir:str, sub_dir:str)->Tuple:
    """return preregistration image as tio Scalar image

    Args:
        data_dir (str): directory in project folder containing the folder of good reconstructions
        sub_dir (str): sub directory of current stack 


    Returns:
        tio.ScalarImage: tio Scalar image of preregstired file
    """
    current_dir = os.getcwd()
    sub_dir_glob = os.path.join(current_dir,data_dir,sub_dir)
    for file in os.listdir(sub_dir_glob):
        if file.endswith("0.nii.gz"):
            pre_reg_tio = tio.ScalarImage(os.path.join(sub_dir_glob,file))
            first_it_path = os.path.join(sub_dir_glob,file)
    return pre_reg_tio, first_it_path

def adjust_size_of_preregistration(pre_reg_tensor:t.tensor, svr_high_res_tensor:t.tensor)->t.tensor:
    """_summary_

    Args:
        pre_reg_tensor (t.tensor): tensor of preregistration
        svr_high_res_tensor (t.tensor): high resolution tensor gained from SVR framework

    Returns:
        t.tensor: pre_reg_tensor in same shape as svr_high_res_tensor
    """
    pre_reg_tensor_shape, svr_high_res_tensor_shape = t.tensor(pre_reg_tensor.shape), t.tensor(svr_high_res_tensor.shape)
    difference_tensor = pre_reg_tensor_shape - svr_high_res_tensor_shape

    #for dimensions where pre_reg is larger than svr_high_res
    cropping_diff =  t.nn.functional.relu(difference_tensor)
    index_array = []
    for index in range(len(cropping_diff)):
        if cropping_diff[index] > 0:
            if index == 0:
                pre_reg_tensor = pre_reg_tensor[:-cropping_diff[index],:,:,:]
            elif index == 1:
                pre_reg_tensor = pre_reg_tensor[:,:-cropping_diff[index],:,:]
            elif index == 2:
                pre_reg_tensor = pre_reg_tensor[:,:,:-cropping_diff[index],:]
            elif index == 3:
                pre_reg_tensor = pre_reg_tensor[:,:,:,:-cropping_diff[index]]
            

    #for dimensions where pre_reg is smaller than svr_high_res
    padding_diff = t.nn.functional.relu(-difference_tensor)
    padding_sequence = (0,padding_diff[-1].item(),0,padding_diff[-2].item(),0,padding_diff[-3].item(),0,padding_diff[-4].item())
    pre_reg_tensor_adjusted = t.nn.functional.pad(pre_reg_tensor,padding_sequence,"constant",0)

    return pre_reg_tensor_adjusted

def pre_registration_data_generation(data_dir:str):
    """
    runs preregistration through multiple directories 
    Args:
        data_dir (str): _description_
    """
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    data_directories = get_list_of_all_dirs(data_dir)
    counter = 0
    for sub_dir in data_directories:
        print(f'Folder {counter}')
        counter = counter + 1

        filenames = get_list_of_stacks(data_dir, sub_dir)
        file_mask = "mask.nii.gz"

        src_folder = os.path.join(data_dir, sub_dir)
        result_folder = src_folder
        prep_folder = "cropped_images"
        tensorboard_path = os.path.join("runs", sub_dir)

        epochs = 1
        inner_epochs = 2 

        pixdim_float = 1.0
        pixdim_list = [pixdim_float] * epochs
        pixdims = [(x,x,x) for x in pixdim_list]

        mode = "bicubic"
        tio_mode = "welch"
        loss_fnc = "ncc"
        opt_alg = "Adam"

        #lr for 2D/3D registratoin
        lr = 0.0013
        
        #lr for volume to volume registration
        lr_vol_vol = 0.0035
        pre_reg_epochs = 18
        #lambda function for setting learning rate
        lambda1 = lambda epoch: [0.1,0.2,0.3,0.4,0.6,0.8,1][epoch] if epoch  < 7  else 1.0

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

        loss_kernel_size = 17

        from_checkpoint = False
        last_rec_file = "reconstruction_volume_10.nii.gz"
        last_epoch = 10
        roi_only = False

        parameter_file = {
            "Result_folder": result_folder,
            "Resolution": pixdim_float,
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
                "Learning_rate_SVR": lr,
                "Learning_rate_prereg": lr_vol_vol,
                "Epochs_preregistration": pre_reg_epochs
                },
            "ROI_only": roi_only,
            "PSF": PSF_doc
        }

        parameter_file_dest = os.path.join(data_dir, sub_dir,"params.json")
        out_file = open(parameter_file_dest, "w")
        json.dump(parameter_file,out_file, indent=6)
        out_file.close()

        svr_optimizer = SVR_optimizer(src_folder, prep_folder, result_folder, filenames, file_mask,pixdims, device, PSF, loss_kernel_size, monai_mode = mode, tio_mode = tio_mode, roi_only=roi_only, lr_vol_vol=lr_vol_vol, tensorboard_path = tensorboard_path, pre_reg_epochs=pre_reg_epochs)
        svr_optimizer.optimize_volume_to_slice(epochs, inner_epochs, lr, PSF, lambda1, loss_fnc=loss_fnc, opt_alg=opt_alg, tensorboard=True, tensorboard_path=tensorboard_path,from_checkpoint=from_checkpoint, last_rec_file=last_rec_file, last_epoch = last_epoch)

        pre_reg_tio, pre_reg_path = get_preregistration(data_dir,sub_dir)
        svr_tio = get_SVR_reconstruction(data_dir,sub_dir)
        first_it_tio, first_it_path = get_first_iteration(data_dir, sub_dir)
        pre_reg_tensor, svr_tensor, first_it_tensor = pre_reg_tio.data, svr_tio.data, first_it_tio.data
        resampled_mask = svr_optimizer.resampled_masks[0]

        #apply mask
        pre_reg_tensor, first_it_tensor = pre_reg_tensor * resampled_mask.data, first_it_tensor * resampled_mask.data
        
        pre_reg_tensor_adjusted, first_it_tensor_adjusted = adjust_size_of_preregistration(pre_reg_tensor, svr_tensor), adjust_size_of_preregistration(first_it_tensor, svr_tensor)
        pre_reg_tio.set_data(pre_reg_tensor_adjusted)
        first_it_tio.set_data(first_it_tensor_adjusted)

        path = os.path.join(os.getcwd(),data_dir,sub_dir,"prereg.nii.gz")
        pre_reg_tio.save(path)

        path = os.path.join(os.getcwd(),data_dir,sub_dir,"first_it.nii.gz")
        first_it_tio.save(path)
        
        #clean up folder
        os.remove(os.path.join(src_folder,"params.json"))
        os.remove(os.path.join(src_folder,"results.csv"))
        os.remove(os.path.join(src_folder,"models_optimizers.pt"))
        os.remove(pre_reg_path)
        os.remove(first_it_path)


