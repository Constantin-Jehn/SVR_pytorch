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
    folder = 'sample_data'
    filename = '10_3T_nody_001.nii.gz'
    pixdim = (2,2,2)
    epochs = 20
    lr = 0.001
    opt_alg = "Adam"
    loss_fn = "ncc"
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    
    svr_opt = SVR_optimizer(folder, filename, pixdim, device)
    target_dict, loss_log = svr_opt.optimize(epochs, lr, loss_fnc =  loss_fn, opt_alg = opt_alg)
    
    plot_title = opt_alg + " pix_dim = (" + ','.join(map(str,(pixdim))) + ")" 
    plt.plot(loss_log)
    plt.title(plot_title)
    plt.xlabel("epoch")
    plt.ylabel(loss_fn)
    plt.grid()
    plt.show()
    
    folder = "test_reconstruction_monai"
    path = os.path.join(folder,opt_alg)
    nifti_saver = monai.data.NiftiSaver(output_dir=path, 
                                        resample = False, mode = "bilinear", padding_mode = "zeros",
                                        separate_folder=False)
    
    save_to = filename[:-7] + '_' + ','.join(map(str,(pixdim))) + '_lr' + str(lr).replace('.',',') + '_' + str(epochs)
    target_dict["image_meta_dict"]["filename_or_obj"] = save_to
    nifti_saver.save(target_dict["image"], meta_data=target_dict["image_meta_dict"])

if __name__ == '__main__':
    optimize()
    #utils.monai_demo()

    #ground_truth, im_slices, target_dict, k = preprocess(folder, filename, pixdim)
    #target_dict, loss_log = optimize(ground_truth, im_slices, target_dict, k)
    
    
    #create randomly rotated slices
    # im_slices = rand_Affine(im_slices)
    
    # rand_vol = utils.reconstruct_3d_volume(im_slices, target_dict)
    # rand_vol["image"] = t.squeeze(rand_vol["image"])
    # add_channel = AddChanneld(keys=["image"])
    # rand_vol = add_channel(rand_vol)
    # resample_to_match = monai.transforms.ResampleToMatch(padding_mode="zeros")
    # rand_vol["image"], rand_vol["image_meta_dict"] = resample_to_match(rand_vol["image"],
    #                                                                           src_meta = target_dict["image_meta_dict"],
    #                                                                           dst_meta = ground_truth["image_meta_dict"])
    # folder = "test_reconstruction_monai"
    # path = os.path.join(folder)
    # nifti_saver = monai.data.NiftiSaver(output_dir=path, output_postfix=".nii.gz", 
    #                                     resample = False, mode = "bilinear", padding_mode = "zeros",
    #                                     separate_folder=False)
    # rand_vol["image_meta_dict"]["filename_or_obj"] = "rand_vol"
    # nifti_saver.save(rand_vol["image"], meta_data=target_dict["image_meta_dict"])