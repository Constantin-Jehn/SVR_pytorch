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
    ToTensord
)
import os
import utils
import numpy as np
import reconstruction_model
import torch as t
from copy import deepcopy

def monai_demo():
    
    mode = "bilinear"
    folder = 'sample_data'
    filename = '10_3T_nody_001.nii.gz'
    path = os.path.join(folder, filename)
    pixdim = (3,3,3)
    
    target_dicts = [{"image": path}]
    loader = LoadImaged(keys = ("image"))
    target_dict = loader(target_dicts[0])
    
    to_tensor = ToTensord(keys = ("image"))
    target_dict = to_tensor(target_dict)
    #ground_pixdim = target_dict["image_meta_dict"]["pixdim"]
    
    add_channel = AddChanneld(keys=["image"])
    target_dict = add_channel(target_dict)
    #adds second "channel for batch"
    #target_dict = add_channel(target_dict)
    #make first dimension the slices
    orientation = monai.transforms.Orientationd(keys = ("image"), axcodes="PLI")
    target_dict = orientation(target_dict)
    
    ground_image, ground_meta = deepcopy(target_dict["image"]), deepcopy(target_dict["image_meta_dict"])
    ground_meta["spatial_shape"] = list(target_dict["image"].shape)[1:]
    
    #resample image to desired pixdim
    mode = "bilinear"
    spacing = Spacingd(keys=["image"], pixdim=pixdim, mode=mode)
    target_dict = spacing(target_dict)
    
    #target_dict = utils.preprocess(target_dict,pixdim)
    im_slices = utils.slices_from_volume(target_dict)
    k = 25
    #how to do rotations
    rotations = t.tensor([0,0,0])
    translations = t.tensor([0,0,0])
    affine_trans = monai.transforms.Affine(rotations.tolist(), translations.tolist())
    im_slices[k]["image"], im_slices[k]["image_meta_dict"]["affine"] = affine_trans(im_slices[k]["image"])
    
    im_slice = im_slices[k]["image"]
    plt.figure("data",(8, 4))
    plt.subplot(1, 3, 1)
    plt.title("next slice")
    plt.imshow(im_slice[0,k+1,:,:], cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("rotated")
    plt.imshow(im_slice[0,k,:,:], cmap="gray")
    plt.subplot(1,3,3)
    plt.title("previous slice")
    plt.imshow(im_slice[0,k-1,:,:], cmap="gray")
    plt.show()
    
    # spatial_size = (84,288,288)
    src_affine = target_dict["image_meta_dict"]["affine"]
    img = target_dict["image"]
    resample_to_match = monai.transforms.ResampleToMatch(padding_mode="zeros")
    resampled_image, resampled_meta = resample_to_match(img,src_meta = target_dict["image_meta_dict"], dst_meta = ground_meta)
    
    k = 12
    plt.figure("data",(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("original_image")
    plt.imshow(ground_image[0,k,:,:], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("resampled")
    plt.imshow(resampled_image[0,k,:,:], cmap="gray")
    plt.show()
    
    
    #save
    folder = "test_reconstruction_monai"
    path = os.path.join(folder)
    
    nifti_saver = monai.data.NiftiSaver(output_dir=path, output_postfix=".nii.gz", 
                                        resample = False, mode = mode, padding_mode = "zeros")
    nifti_saver.save(target_dict["image"], meta_data=target_dict["image_meta_dict"])

def optimize():
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    mode = "bilinear"
    folder = 'sample_data'
    filename = '10_3T_nody_001.nii.gz'
    path = os.path.join(folder, filename)
    pixdim = (4,4,4)
    
    target_dicts = [{"image": path}]
    loader = LoadImaged(keys = ("image"))
    target_dict = loader(target_dicts[0])
    
    to_tensor = ToTensord(keys = ("image"))
    target_dict = to_tensor(target_dict)
    #ground_pixdim = target_dict["image_meta_dict"]["pixdim"]
    
    add_channel = AddChanneld(keys=["image"])
    target_dict = add_channel(target_dict)
    
    #make first dimension the slices
    orientation = monai.transforms.Orientationd(keys = ("image"), axcodes="PLI")
    target_dict = orientation(target_dict)
    
    #save initial images for loss function
    ground_image, ground_meta = deepcopy(target_dict["image"]), deepcopy(target_dict["image_meta_dict"])
    ground_meta["spatial_shape"] = list(target_dict["image"].shape)[1:]
    ground_truth = {"image": ground_image,
                    "image_meta_data": ground_meta}
    ground_truth = add_channel(ground_truth)
    
    #resample image to desired pixdim of reconstruction volume
    mode = "bilinear"
    spacing = Spacingd(keys=["image"], pixdim=pixdim, mode=mode)
    target_dict = spacing(target_dict)
    
    target_dict = add_channel(target_dict)

    im_slices = utils.slices_from_volume(target_dict)
    k = len(im_slices)
    
    model = reconstruction_model.ReconstructionMonai(k,device)
    monai_ncc = monai.losses.LocalNormalizedCrossCorrelationLoss(spatial_dims=3)
    optimizer = t.optim.SGD(model.parameters(), lr = 0.01)
    
    ground_spatial_dim = ground_truth["image_meta_data"]["spatial_shape"]
    resample_to_match = monai.transforms.ResampleToMatch(padding_mode="zeros")
    
    tgt_meta = deepcopy(target_dict["image_meta_dict"])
    
    for epoch in range(0,2):
        model.train()
        optimizer.zero_grad()
        #make prediction
        target_dict = model(im_slices, target_dict, ground_spatial_dim)
        
        #bring target into image-shape for resampling
        target_dict["image"] = t.squeeze(target_dict["image"])
        target_dict = add_channel(target_dict)
        #resample for loss
        target_dict["image"], target_dict["image_meta_dict"] = resample_to_match(target_dict["image"],
                                                                                 src_meta = tgt_meta,
                                                                                 dst_meta = ground_meta)
        #bring target into batch-shape
        target_dict = add_channel(target_dict)
        
        print("target in high res")
        plt.imshow(t.squeeze(target_dict["image"])[12,:,:].detach().numpy(), cmap="gray")
        plt.show()
        #img = target_dict["image"]
        loss = monai_ncc(target_dict["image"], ground_truth["image"])
        t.autograd.set_detect_anomaly(True)
        loss.backward(retain_graph=True)
        optimizer.step()
        #target_dict = spacing(target_dict)
        print(f'Epoch: {epoch} Loss: {loss}')
    
    #bring target into image shape
    target_dict["image"] = t.squeeze(target_dict["image"])
    target_dict= add_channel(target_dict)
    return target_dict, ground_meta


if __name__ == '__main__':
    #monai_demo()
    target_dict, ground_meta = optimize()
    folder = "test_reconstruction_monai"
    path = os.path.join(folder)
    nifti_saver = monai.data.NiftiSaver(output_dir=path, output_postfix=".nii.gz", 
                                        resample = False, mode = "bilinear", padding_mode = "zeros",
                                        separate_folder=False)
    target_dict["image_meta_dict"]["filename_or_obj"] = "opt_reconstr"
    nifti_saver.save(target_dict["image"], meta_data=target_dict["image_meta_dict"])



    