import torch as t
import monai
from monai.transforms import (
    AddChanneld,
    LoadImaged,
    ToTensord,
)
import torchio as tio
import torchmetrics as tm
import os
import numpy as np
import json

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

if __name__ == '__main__':
    #set category to evaluate prereg, image0 or all
    category = "prereg"
    #generate folder paths
    folder_cycle, folder_labels, folder_images = create_path(category, "CycleGAN"), create_path(category, "label"), create_path(category, "image")
    #get sorted file lists
    files_cycle, files_labels, files_images = get_sorted_file_list (folder_cycle), get_sorted_file_list(folder_labels), get_sorted_file_list(folder_images)
    assert len(files_cycle) == len(files_labels) or len(files_cycle) == len(files_images), f"Evaluation folder should contain same number of files got {len(files_cycle)} CycleGAN, {len(files_labels)} labels and {len(files_images)} images"
    n_files = len(files_cycle)

    #define similarity metrics
    tm_ssim = tm.StructuralSimilarityIndexMeasure(kernel_size=99, reduction='sum')
    monai_psnr = monai.metrics.PSNRMetric(1.0)

    #def datstructure to store results
    #define lists to store values
    image_to_label_psnr, image_to_label_ssim, cycle_to_label_psnr, cycle_to_label_ssim = [], [], [], []

    for n in range(0,n_files):
        #in sorted list the the files should match -> check by first 2 elements
        cycle_id, label_id, image_id = files_cycle[n][:2], files_labels[n][:2], files_images[n][:2]
        assert cycle_id == image_id or cycle_id == label_id, f"Comparing file ids should match, but got cycle_id: {cycle_id}, label_id: {label_id}, image_id: {image_id}"

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
        image_to_label_psnr.append(monai_psnr(tensor_image, tensor_label).item())
        image_to_label_ssim.append(tm_ssim(tensor_image, tensor_label).item())

        cycle_to_label_psnr.append(monai_psnr(tensor_cycle, tensor_label).item())
        cycle_to_label_ssim.append(tm_ssim(tensor_cycle,tensor_label).item())

    image_to_label_psnr_mean, image_to_label_psnr_std = np.mean(np.array(image_to_label_psnr)), np.std(np.array(image_to_label_psnr))
    image_to_label_ssim_mean, image_to_label_ssim_std = np.mean(np.array(image_to_label_ssim)), np.std(np.array(image_to_label_ssim))

    cycle_to_label_psnr_mean, cycle_to_label_psnr_std = np.mean(np.array(cycle_to_label_psnr)), np.std(np.array(cycle_to_label_psnr))
    cycle_to_label_ssim_mean, cycle_to_label_ssim_std = np.mean(np.array(cycle_to_label_ssim)), np.std(np.array(cycle_to_label_ssim))

    results = {
            'image_to_label':{
                'PSNR': {
                    'mean': image_to_label_psnr_mean,
                    'std': image_to_label_psnr_std,
                    'values':image_to_label_psnr    
                },
                'SSIM': {
                    'mean': image_to_label_ssim_mean,
                    'std': image_to_label_ssim_std,
                    'values':image_to_label_ssim    
                } 
            },
            'cycle_to_label':{
                'PSNR':{
                    'mean': cycle_to_label_psnr_mean,
                    'std': cycle_to_label_psnr_std,
                    'values': cycle_to_label_psnr
                },
                'SSIM': {
                    'mean': cycle_to_label_ssim_mean,
                    'std': cycle_to_label_ssim_std,
                    'values': cycle_to_label_ssim    
                } 
            }
        }
    
    result_file_dest = os.path.join(os.getcwd(), category, "metrics.json")
    out_file = open(result_file_dest, "w")
    json.dump(results,out_file, indent=6)
    out_file.close()










    
