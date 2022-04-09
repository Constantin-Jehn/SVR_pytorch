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
import utils
import numpy as np
import reconstruction_model
import torch as t
from copy import deepcopy
from SVR_optimizer import SVR_optimizer

def optimize():
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    
    folder = 'sample_data'
    filename = '10_3T_nody_001_cropped.nii'
    pixdim = (3,3,3)
    epochs = 10
    lr = 0.001
    opt_alg = "Adam"
    loss_fn = "ncc"
    mode = "bilinear"
    save_to = filename[:-7] + '_' + ','.join(map(str,(pixdim))) + '_lr' + str(lr).replace('.',',') + '_' + str(epochs) + '_' + mode

    svr_opt = SVR_optimizer(folder, filename, pixdim, device, mode)
    target_dict, loss_log = svr_opt.optimize(epochs, lr, loss_fnc =  loss_fn, opt_alg = opt_alg)
    
    
    plot_dest = os.path.join("plots", save_to)
    plot_title = opt_alg + " pix_dim = (" + ','.join(map(str,(pixdim))) + ")" 
    plt.plot(loss_log)
    plt.title(plot_title)
    plt.xlabel("epoch")
    plt.ylabel(loss_fn)
    plt.grid()
    plt.savefig(plot_dest)
    plt.show()
    
    folder = "test_reconstruction_monai"
    path = os.path.join(folder,opt_alg)
    nifti_saver = monai.data.NiftiSaver(output_dir=path, 
                                        resample = False, mode = mode, padding_mode = "zeros",
                                        separate_folder=False)
    
    target_dict["image_meta_dict"]["filename_or_obj"] = save_to
    nifti_saver.save(target_dict["image"], meta_data=target_dict["image_meta_dict"])

if __name__ == '__main__':
    filenames = ["10_3T_nody_001.nii.gz",
    "10_3T_nody_002.nii.gz",
    "14_3T_nody_001.nii.gz",
    "14_3T_nody_002.nii.gz",
    "21_3T_nody_001.nii.gz",
    "21_3T_nody_002.nii.gz",
    "23_3T_nody_001.nii.gz",
    "23_3T_nody_002.nii.gz"]
    file_mask = "mask_10_3T_brain_smooth.nii.gz"
    file_world = "world.nii.gz"
    pixdim = (1.0, 1.0, 1.0)

    src_folder = "sample_data"
    dst_folder = "cropped_images"
    src_folder = "sample_data"
    dst_folder = "cropped_images"
    
    svr_optimizer = SVR_optimizer(src_folder,dst_folder, filenames, file_mask,pixdim, "cpu", mode = "bilinear")
    svr_optimizer.optimize_multiple_stacks(1, 0.1)
    #optimize()
