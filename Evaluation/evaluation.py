import torch as t
import monai
from monai.transforms import (
    AddChanneld,
    LoadImaged,
    ToTensord,
)
import torchio as tio
import torchmetrics as tm
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
import os
import numpy as np
import json
import pandas as pd

def create_path(category: str, folder_name:str)->str:
    return os.path.join(os.getcwd(), category, folder_name)

def get_sorted_file_list(dir:str)->list:
    list_of_files = sorted(os.listdir(dir))
    for file in list_of_files:
        if file[0] == '.':
            list_of_files.remove(file)
    return list_of_files

def normalize_to_unit_interval(tio_image:tio.ScalarImage):
    min_val, max_val = t.min(tio_image.data), t.max(tio_image.data)
    range_val = max_val - min_val
    norm_data = t.div((tio_image.data - min_val), range_val)
    tio_image.set_data(norm_data)
    return tio_image

def write_in_overview_file(file_path:str, category, lr, image_to_label_mean, image_to_label_std, image_to_label_sem, cycle_to_label_mean, cycle_to_label_std, cycle_to_label_sem):
    
    json_obj = open(file_path)
    PSNR_json = json.load(json_obj)

    PSNR_json[category]["image_to_label_means"][lr] = image_to_label_mean
    PSNR_json[category]["image_to_label_std"][lr] = image_to_label_std
    PSNR_json[category]["image_to_label_sem"][lr] = image_to_label_sem

    PSNR_json[category]["cycle_to_label_means"][lr] = cycle_to_label_mean
    PSNR_json[category]["cylce_to_label_std"][lr] = cycle_to_label_std
    PSNR_json[category]["cycle_to_label_sem"][lr] = cycle_to_label_sem

    out_file = open(file_path, "w")
    json.dump(PSNR_json,out_file, indent=6)
    out_file.close()

def get_mean_std_sem(input_array, n_files):
    mean_val = np.mean(np.array(input_array))
    std_val = np.std(np.array(input_array))
    sem_val = std_val / np.sqrt(n_files)
    return mean_val, std_val, sem_val

def nmse(input_tensor, target_tensor):
    difference = input_tensor-target_tensor
    nominator = t.pow(t.norm(difference,p = 'fro'),2)
    denominator = t.pow(t.norm(input_tensor, p = 'fro'),2)
    nmse = t.div(nominator, denominator).item()
    return nmse

def lpip_input_format(image_2d:t.tensor):
    #bring to [-1,1] range
    image_2d = t.mul(image_2d,2) - 1
    #formate [N,3,H,W]
    return image_2d.repeat(3,1,1).unsqueeze(0)

def lpip_over_volume(input_tensor,target_tensor,dimension):
    n = input_tensor.shape[dimension]
    lpip_values = []
    for i in range(0,n):
        if dimension == 0:
            input_2d, target_2d = input_tensor[i,:,:], target_tensor[i,:,:]
        if dimension == 1:
            input_2d, target_2d = input_tensor[:,i,:], target_tensor[:,i,:]
        if dimension == 2:
            input_2d, target_2d = input_tensor[:,:,i], target_tensor[:,:,i]
        if i == 0:
            input_stack, target_stack =  input_2d.repeat(3,1,1).unsqueeze(0), target_2d.repeat(3,1,1).unsqueeze(0)
        else:
            input_stack, target_stack = t.cat((input_stack,input_2d.repeat(3,1,1).unsqueeze(0))), t.cat((target_stack,target_2d.repeat(3,1,1).unsqueeze(0)))
    return input_stack, target_stack


def lpips(input_tensor, target_tensor):
    #lpip expect 3 channel image
    input_tensor, target_tensor = input_tensor.squeeze(), target_tensor.squeeze()
    #get middle dimension to extract 2d images
    tensor_shapes = input_tensor.shape
    dim_0_mid, dim_1_mid, dim_2_mid = int(t.round(t.div(tensor_shapes[0],2)).item()), int(t.round(t.div(tensor_shapes[1],2)).item()),int(t.round(t.div(tensor_shapes[2],2)).item())

    delta = 2
    input_image_0, input_image_1, input_image_2 = input_tensor[dim_0_mid-delta:dim_0_mid + delta,:,:], input_tensor[:,dim_1_mid-delta:dim_1_mid+delta,:], input_tensor[:,:, dim_2_mid-delta:dim_2_mid+delta]
    
    target_image_0, target_image_1, target_image_2 = target_tensor[dim_0_mid-delta:dim_0_mid+delta,:,:], target_tensor[:,dim_1_mid-delta:dim_1_mid+delta,:], target_tensor[:,:,dim_2_mid-delta:dim_2_mid+delta]
    
    """
    input_image_0, input_image_1, input_image_2 = lpip_input_format(input_image_0), lpip_input_format(input_image_1), lpip_input_format(input_image_2)
    target_image_0, target_image_1, target_image2 = lpip_input_format(target_image_0), lpip_input_format(target_image_1), lpip_input_format(target_image2)
    """
    input_image_0, target_image_0 = lpip_over_volume(input_image_0, target_image_0,0)
    input_image_1, target_image_1 = lpip_over_volume(input_image_1, target_image_1,1)
    input_image_2, target_image_2 = lpip_over_volume(input_image_2, target_image_2,2)

    lpip_values_dim_0, lpip_values_dim_1, lpip_values_dim_2 = lpips_metric(input_image_0, target_image_0), lpips_metric(input_image_1, target_image_1), lpips_metric(input_image_2, target_image_2)
    result = t.mean(t.tensor([lpip_values_dim_0, lpip_values_dim_1, lpip_values_dim_2])).item()
    
    return result


if __name__ == '__main__':
    #set learning rate to store data in overview
    lr_list = ["lr=0.001", "lr=0.0004", "lr=0.0002"]
    
    base_path_list = ["/Users/constantin/Documents/05_FAU_CE/4.Semester/Msc/Cycle_GAN_results/test_lr_0.001_cross_1_27_09",
    "/Users/constantin/Documents/05_FAU_CE/4.Semester/Msc/Cycle_GAN_results/test_lr_0.0004_19_09",
    "/Users/constantin/Documents/05_FAU_CE/4.Semester/Msc/Cycle_GAN_results/test_lr_0.0002_19_09",
    ]

    category_list = ["test_all", "test_prereg","test_image0"]
    
    PSNR_overview_path = "/Users/constantin/Documents/05_FAU_CE/4.Semester/Msc/Cycle_GAN_results/PSNR_results_all.json"
    SSIM_overview_path =  "/Users/constantin/Documents/05_FAU_CE/4.Semester/Msc/Cycle_GAN_results/SSIM_results_all.json"
    NCC_overview_path =  "/Users/constantin/Documents/05_FAU_CE/4.Semester/Msc/Cycle_GAN_results/NCC_results_all.json"
    NMSE_overview_path =  "/Users/constantin/Documents/05_FAU_CE/4.Semester/Msc/Cycle_GAN_results/NMSE_results_all.json"
    LPIPS_overview_path =  "/Users/constantin/Documents/05_FAU_CE/4.Semester/Msc/Cycle_GAN_results/LPIPS_results_all_20.json"
    
    #set category to evaluate test_prereg, test_image0 or test_all
    for i in range(0,len(lr_list)):
        lr = lr_list[i]
        base_path = base_path_list[i]

        for category in category_list:

            #generate folder paths
            folder_cycle, folder_labels, folder_images = os.path.join(base_path, category, "CycleGAN"), os.path.join(base_path, category, "labels"), os.path.join(base_path, category, "images")
            #get sorted file lists
            files_cycle, files_labels, files_images = get_sorted_file_list (folder_cycle), get_sorted_file_list(folder_labels), get_sorted_file_list(folder_images)
            assert len(files_cycle) == len(files_labels) and len(files_cycle) == len(files_images), f"Evaluation folder should contain same number of files got {len(files_cycle)} CycleGAN, {len(files_labels)} labels and {len(files_images)} images"
            n_files = len(files_cycle)

            #define similarity metrics
            tm_ssim = tm.StructuralSimilarityIndexMeasure(kernel_size=9, reduction='sum')
            monai_psnr = monai.metrics.PSNRMetric(1.0)
            monai_ncc = monai.losses.LocalNormalizedCrossCorrelationLoss(spatial_dims = 3, kernel_size = 9)

            #def datstructure to store results
            #define lists to store values
            image_to_label_psnr, image_to_label_ssim, cycle_to_label_psnr, cycle_to_label_ssim = [], [], [], []
            image_to_label_ncc, cycle_to_label_ncc, image_to_label_nmse, cycle_to_label_nmse = [],[],[], []
            image_to_label_lpips, cycle_to_label_lpips = [],[]

            for n in range(0,n_files):
                #in sorted list the the files should match -> check by first 2 elements
                cycle_id, label_id, image_id = files_cycle[n][:2], files_labels[n][:2], files_images[n][:2]
                assert cycle_id == image_id and cycle_id == label_id, f"Comparing file ids should match, but got cycle_id: {cycle_id}, label_id: {label_id}, image_id: {image_id}"

                #get absolute paths to current images
                cylce_path, label_path, image_path = os.path.join(folder_cycle,files_cycle[n]), os.path.join(folder_labels, files_labels[n]), os.path.join(folder_images, files_images[n])

                #open the files as tio ScalarImage
                tio_cycle, tio_label, tio_image = tio.ScalarImage(cylce_path), tio.ScalarImage(label_path), tio.ScalarImage(image_path)

                #sample all images to cycle's coordinates
                resampler = tio.Resample(tio_cycle)
                tio_label, tio_image = resampler(tio_label), resampler(tio_image)

                #normalize all iamges to unit interval and get data tensors
                tio_cycle, tio_image, tio_label = normalize_to_unit_interval(tio_cycle), normalize_to_unit_interval(tio_image), normalize_to_unit_interval(tio_label)
                tensor_cycle, tensor_image, tensor_label = tio_cycle.data.float(), tio_image.data.float(), tio_label.data.float()
                
                #fill metric lists
                """
                image_to_label_psnr.append(monai_psnr(tensor_image, tensor_label).item())
                image_to_label_ssim.append(tm_ssim(tensor_image, tensor_label).item())
                image_to_label_ncc.append(monai_ncc(tensor_image.unsqueeze(0), tensor_label.unsqueeze(0)).item())
                image_to_label_nmse.append(nmse(tensor_image, tensor_label))
                """
                image_to_label_lpips.append(lpips(tensor_image, tensor_label))

                """
                cycle_to_label_psnr.append(monai_psnr(tensor_cycle, tensor_label).item())
                cycle_to_label_ssim.append(tm_ssim(tensor_cycle,tensor_label).item())
                cycle_to_label_ncc.append(monai_ncc(tensor_cycle.unsqueeze(0), tensor_label.unsqueeze(0)).item())
                cycle_to_label_nmse.append(nmse(tensor_cycle, tensor_label))
                """
                cycle_to_label_lpips.append(lpips(tensor_cycle, tensor_label))


            #aggregate means, std, sem
            """
            image_to_label_psnr_mean, image_to_label_psnr_std, image_to_label_psnr_sem = get_mean_std_sem(image_to_label_psnr, n_files)
            image_to_label_ssim_mean, image_to_label_ssim_std, image_to_label_ssim_sem = get_mean_std_sem(image_to_label_ssim, n_files)
            image_to_label_ncc_mean, image_to_label_ncc_std, image_to_label_ncc_sem = get_mean_std_sem(image_to_label_ncc, n_files)
            image_to_label_nmse_mean, image_to_label_nmse_std, image_to_label_nmse_sem = get_mean_std_sem(image_to_label_nmse, n_files)
            """
            image_to_label_lpips_mean, image_to_label_lpips_std, image_to_label_lpips_sem = get_mean_std_sem(image_to_label_lpips, n_files)

            """
            cycle_to_label_psnr_mean, cycle_to_label_psnr_std, cycle_to_label_psnr_sem = get_mean_std_sem(cycle_to_label_psnr, n_files)
            cycle_to_label_ssim_mean, cycle_to_label_ssim_std, cycle_to_label_ssim_sem = get_mean_std_sem(cycle_to_label_ssim, n_files)
            cycle_to_label_ncc_mean, cycle_to_label_ncc_std, cycle_to_label_ncc_sem = get_mean_std_sem(cycle_to_label_ncc, n_files)
            cycle_to_label_nmse_mean, cycle_to_label_nmse_std, cycle_to_label_nmse_sem = get_mean_std_sem(cycle_to_label_nmse, n_files)
            """
            cycle_to_label_lpips_mean, cycle_to_label_lpips_std, cycle_to_label_lpips_sem = get_mean_std_sem(cycle_to_label_lpips, n_files)

            """
            results = {
                    'path': os.path.join(base_path, category),
                    'image_to_label':{
                        'PSNR': {
                            'mean': image_to_label_psnr_mean,
                            'std': image_to_label_psnr_std,
                            'sem': image_to_label_psnr_sem,
                            'values':image_to_label_psnr    
                        },
                        'SSIM': {
                            'mean': image_to_label_ssim_mean,
                            'std': image_to_label_ssim_std,
                            'sem': image_to_label_ssim_sem,
                            'values':image_to_label_ssim    
                        },
                        'NCC': {
                            'mean': image_to_label_ncc_mean,
                            'std': image_to_label_ncc_std,
                            'sem': image_to_label_ncc_sem,
                            'values':image_to_label_ncc    
                        },
                        'NMSE': {
                            'mean': image_to_label_nmse_mean,
                            'std': image_to_label_nmse_std,
                            'sem': image_to_label_nmse_sem,
                            'values':image_to_label_nmse    
                        },
                        'LPIPS': {
                            'mean': image_to_label_lpips_mean,
                            'std': image_to_label_lpips_std,
                            'sem': image_to_label_lpips_sem,
                            'values':image_to_label_lpips    
                        }
                    },
                    'cycle_to_label':{
                        'PSNR':{
                            'mean': cycle_to_label_psnr_mean,
                            'std': cycle_to_label_psnr_std,
                            'sem': cycle_to_label_psnr_sem,
                            'values': cycle_to_label_psnr
                        },
                        'SSIM': {
                            'mean': cycle_to_label_ssim_mean,
                            'std': cycle_to_label_ssim_std,
                            'sem': cycle_to_label_ssim_sem,
                            'values': cycle_to_label_ssim    
                        },
                        'NCC': {
                            'mean': cycle_to_label_ncc_mean,
                            'std': cycle_to_label_ncc_std,
                            'sem': cycle_to_label_ncc_sem,
                            'values': cycle_to_label_ncc    
                        },
                        'NCC': {
                            'mean': cycle_to_label_nmse_mean,
                            'std': cycle_to_label_nmse_std,
                            'sem': cycle_to_label_nmse_sem,
                            'values': cycle_to_label_nmse    
                        },
                        'LPIPS': {
                            'mean': cycle_to_label_lpips_mean,
                            'std': cycle_to_label_lpips_std,
                            'sem': cycle_to_label_lpips_sem,
                            'values':cycle_to_label_lpips    
                        }
                    }
                }
            
            result_file_dest = os.path.join(base_path, category, category + "_metrics.json")
            out_file = open(result_file_dest, "w")
            json.dump(results,out_file, indent=6)
            out_file.close()
            """
            
            """
            write_in_overview_file(PSNR_overview_path, category, lr, image_to_label_psnr_mean, image_to_label_psnr_std, image_to_label_psnr_sem, cycle_to_label_psnr_mean, cycle_to_label_psnr_std, cycle_to_label_psnr_sem)
            write_in_overview_file(SSIM_overview_path,category, lr, image_to_label_ssim_mean, image_to_label_ssim_std, image_to_label_ssim_sem, cycle_to_label_ssim_mean, cycle_to_label_ssim_std, cycle_to_label_ssim_sem)
            write_in_overview_file(NCC_overview_path,category, lr, image_to_label_ncc_mean, image_to_label_ncc_std, image_to_label_ncc_sem, cycle_to_label_ncc_mean, cycle_to_label_ncc_std, cycle_to_label_ncc_sem)
            write_in_overview_file(NMSE_overview_path,category, lr, image_to_label_nmse_mean, image_to_label_nmse_std, image_to_label_nmse_sem, cycle_to_label_nmse_mean, cycle_to_label_nmse_std, cycle_to_label_nmse_sem)
            """
            write_in_overview_file(LPIPS_overview_path,category, lr, image_to_label_lpips_mean, image_to_label_lpips_std, image_to_label_lpips_sem, cycle_to_label_lpips_mean, cycle_to_label_lpips_std, cycle_to_label_lpips_sem)

            print(f'category: {category}, : {lr_list[i]}')





    
