import os
import random
import shutil

from torch import rand

if __name__ == '__main__':
    source_folder = os.path.join(os.getcwd(),"Good")

    indices_shuffled = list(range(220))
    random.shuffle(indices_shuffled)

    list_of_source_folders = [dir for dir in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder,dir))]
    list_of_source_folders = sorted(list_of_source_folders)

    dst_folder =  os.path.join(os.getcwd(),"Dataset_06_09")
    counter = 0
    for dir in list_of_source_folders:
        current_folder = os.path.join(source_folder,dir)

        SVRTK_src = os.path.join(current_folder,"outputSVR.nii.gz")
        SVRTK_filename = str(indices_shuffled[counter]) + "_label" +  ".nii.gz"
        SVRTK_dst = os.path.join(dst_folder,"label")
        shutil.copy(SVRTK_src, SVRTK_dst)
        os.rename(os.path.join(SVRTK_dst,"outputSVR.nii.gz"),os.path.join(SVRTK_dst,SVRTK_filename))

        
        image0_src = os.path.join(current_folder,"SVR_reco","image0.nii.gz")
        image0_filename = str(indices_shuffled[counter]) + "_image0" +  ".nii.gz"
        image0_dst = os.path.join(dst_folder, "image0_back_up")
        shutil.copy(image0_src, image0_dst)
        os.rename(os.path.join(image0_dst,"image0.nii.gz"),os.path.join(image0_dst, image0_filename))
        
        
        prereg_src = os.path.join(current_folder,"prereg.nii.gz")
        prereg_filename = str(indices_shuffled[counter]) + "_prereg" + ".nii.gz"
        prereg_dst = os.path.join(dst_folder, "image")
        shutil.copy(prereg_src, prereg_dst)
        os.rename(os.path.join(prereg_dst,"prereg.nii.gz"),os.path.join(prereg_dst, prereg_filename))

        counter = counter + 1

    #now files where there is no preregistration
    source_folder = os.path.join(os.getcwd(),"Bad+Middle")

    list_of_source_folders = [dir for dir in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder,dir))]
    list_of_source_folders = sorted(list_of_source_folders)

    for dir in list_of_source_folders:
        current_folder = os.path.join(source_folder,dir)

        SVRTK_src = os.path.join(current_folder,"outputSVR.nii.gz")
        SVRTK_filename = str(indices_shuffled[counter]) + "_label" +  ".nii.gz"
        SVRTK_dst = os.path.join(dst_folder,"label")
        shutil.copy(SVRTK_src, SVRTK_dst)
        os.rename(os.path.join(SVRTK_dst,"outputSVR.nii.gz"),os.path.join(SVRTK_dst,SVRTK_filename))

        image0_src = os.path.join(current_folder,"SVR_reco","image0.nii.gz")
        image0_filename = str(indices_shuffled[counter]) + "_image0" +  ".nii.gz"
        image0_dst = os.path.join(dst_folder, "image")
        shutil.copy(image0_src, image0_dst)
        os.rename(os.path.join(image0_dst,"image0.nii.gz"),os.path.join(image0_dst, image0_filename))

        counter = counter + 1


    



