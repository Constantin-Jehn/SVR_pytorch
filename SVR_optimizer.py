import torchio as tio
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


class SVR_optimizer():
    def __init__(self,src_folder, prep_folder, stack_filenames, mask_filename, pixdim, device, mode):
        """
        constructer of SVR_optimizer class
        Parameters
        ----------
        src_folder : string
            initial nifti_files
        prep_folder : string
            folder to save prepocessed files
        stack_filenames : list
            of filenames of stacks to be reconstructed
        mask_filename : string
            nifti filename to crop input images
        pixdim : list
            DESCRIPTION.
        device : TYPE
            DESCRIPTION.
        mode : string
            interpolation mode

        Returns
        -------
        None.

        """
        self.src_folder = src_folder
        self.prep_folder = prep_folder
        self.stack_filenames = stack_filenames
        self.mask_filename = mask_filename
        self.pixdim = pixdim
        self.device = device
        self.mode = mode
        
        self.crop_images()
        self.ground_truth = self.load_stacks()
        self.resample_to_pixdim()
        self.stacks = self.load_stacks()
        #self.ground_truth, self.im_slices, self.target_dict, self.k = self.preprocess()


    def crop_images(self):
        """
        Parameters
        ----------
        src_folder : string
            folder of images to be cropped
        filenames : string
            stacks that should be cropped
        mask_filename : string
            mask for cropping
        dst_folder : string
            folder to save cropped files

        Returns
        -------
        None.
        """
        path_mask = os.path.join(self.src_folder, self.mask_filename)
        mask = tio.LabelMap(path_mask)
        path_dst = os.path.join(self.prep_folder, self.mask_filename)
        mask.save(path_dst)
        k = len(self.stack_filenames)
        for i in range(0,k):
            filename = self.stack_filenames[i]
            path_stack = os.path.join(self.src_folder, filename)
            stack = tio.ScalarImage(path_stack)
            resampler = tio.transforms.Resample(stack)
            resampled_mask = resampler(deepcopy(mask))
            subject = tio.Subject(stack = stack, mask = resampled_mask)
            
            masked_indices_1 = t.nonzero(subject.mask.data)
            min_indices = np.array([t.min(masked_indices_1[:,1]).item(), t.min(masked_indices_1[:,2]).item(),t.min(masked_indices_1[:,3]).item()])
            max_indices = np.array([t.max(masked_indices_1[:,1]).item(), t.max(masked_indices_1[:,2]).item(),t.max(masked_indices_1[:,3]).item()])
            roi_size = (max_indices - min_indices) 
            
            cropper = tio.CropOrPad(list(roi_size),mask_name= 'mask')
            
            cropped_stack = cropper(subject)
            path_dst = os.path.join(self.prep_folder, filename)
            cropped_stack.stack.save(path_dst)
        #create a common place for the reconstruction
        common_image = tio.Image(tensor = t.zeros((1,100,100,100)))
        path_dst = os.path.join(self.prep_folder, "world.nii.gz")
        common_image.save(path_dst)
    
    
    def load_stacks(self):
        """
        After cropping the initial images in low resolution are saved in their original coordinates
        for the loss computation
        Returns
        -------
        ground_truth : list
            list of dictionaries containing the ground truths

        """
        add_channel = AddChanneld(keys=["image"])
        loader = LoadImaged(keys = ["image"])
        to_tensor = ToTensord(keys = ["image"])
        k = len(self.stack_filenames)
        stack_list = list()
        for i in range(0,k):
            path = os.path.join(self.prep_folder, self.stack_filenames[i])
            stack_dict = {"image": path}
            stack_dict = loader(stack_dict)
            stack_dict = to_tensor(stack_dict)
            stack_dict = add_channel(stack_dict)
            #keep meta data correct
            stack_dict["image_meta_dict"]["spatial_shape"] = np.array(list(stack_dict["image"].shape)[1:])
            stack_list.append(stack_dict)
        return stack_list
            

    def resample_to_pixdim(self):
        """
        After cropping and having saved the ground truth image, bring the images
        to desired resolution for reconstruction

        Returns
        -------
        None.

        """
        resampler = tio.transforms.Resample(self.pixdim)
        k = len(self.stack_filenames)
        for i in range(0,k):
            filename = self.stack_filenames[i]
            path_stack = os.path.join(self.prep_folder, filename)
            stack = tio.ScalarImage(path_stack)
            resampled_stack = resampler(stack)
            path_dst = os.path.join(self.prep_folder, filename)
            resampled_stack.save(path_dst)    
    
    def preprocess(self):
        """
        Parameters
        ----------
        folder : string
            folder of nifti file to be processed
        filename : string
            filename of nifti file to be processed
        pixdim : TYPE
            DESCRIPTION.

        Returns
        -------
        ground_truth : dictionary
            dictionary with the basic image slices as reference for loss function
        im_slices : list of dictionaries
            contains all slices represented in 3d space
        target_dict : dictionary
            DESCRIPTION.
        k : int
            number of slices

        """
        mode = self.mode
        path = os.path.join(self.folder, self.filename)
        
        add_channel = AddChanneld(keys=["image"])
        orientation = monai.transforms.Orientationd(keys = ("image"), axcodes="PLI")
        spacing = Spacingd(keys=["image"], pixdim=self.pixdim, mode=mode)
        
        target_dicts = [{"image": path}]
        loader = LoadImaged(keys = ("image"))
        target_dict = loader(target_dicts[0])
        
        to_tensor = ToTensord(keys = ("image"))
        target_dict = to_tensor(target_dict)
        #ground_pixdim = target_dict["image_meta_dict"]["pixdim"]
        target_dict = add_channel(target_dict)
        
        #make first dimension the slices
        target_dict = orientation(target_dict)
        
        #save initial images for loss function
        ground_image, ground_meta = deepcopy(target_dict["image"]), deepcopy(target_dict["image_meta_dict"])
        ground_meta["spatial_shape"] = list(target_dict["image"].shape)[1:]
        ground_truth = {"image": ground_image,
                        "image_meta_dict": ground_meta}
        ground_truth = add_channel(ground_truth)
        
        #resample image to desired pixdim of reconstruction volume
        mode = "bilinear"
        target_dict = spacing(target_dict)
        target_dict = add_channel(target_dict)
        im_slices = utils.slices_from_volume(target_dict)
        k = len(im_slices)
        return ground_truth, im_slices, target_dict, k
    
    
    def optimize(self, epochs, lr, loss_fnc = "ncc", opt_alg = "SGD"):
        """
        Parameters
        ----------
        ground_truth : dictionary
            contains initial images and meta_data
        im_slices : list
            contains all slices as dictionaries containing 3d representation and meta data
        target_dict : dictionary
            DESCRIPTION.
        k : TYPE
            DESCRIPTION.

        Returns
        -------
        target_dict : dictionary
            containing reconstructed volume

        """
        add_channel = AddChanneld(keys=["image"])
        
        model = reconstruction_model.ReconstructionMonai(self.k,self.device, self.mode)
        model.to(self.device)
        if opt_alg == "SGD":
            optimizer = t.optim.SGD(model.parameters(), lr = lr)
        elif(opt_alg == "Adam"):
            optimizer = t.optim.Adam(model.parameters(), lr = lr)
        else:
            assert("Choose SGD or Adam as optimizer")
        
        ground_spatial_dim = self.ground_truth["image_meta_dict"]["spatial_shape"]
        resample_to_match = monai.transforms.ResampleToMatch(padding_mode="zeros")
        
        tgt_meta = deepcopy(self.target_dict["image_meta_dict"])
        ground_meta = self.ground_truth["image_meta_dict"]
        
        loss_log = list()
        for epoch in range(0,epochs):
            model.train()
            optimizer.zero_grad()
            #make prediction
            self.target_dict = model(self.im_slices, self.target_dict, ground_spatial_dim)
            
            #bring target into image-shape for resampling
            self.target_dict["image"] = t.squeeze(self.target_dict["image"])
            self.target_dict = add_channel(self.target_dict)
            #resample for loss
            self.target_dict["image"], self.target_dict["image_meta_dict"] = resample_to_match(self.target_dict["image"],
                                                                                     src_meta = tgt_meta,
                                                                                     dst_meta = ground_meta)
            #bring target into batch-shape
            self.target_dict = add_channel(self.target_dict)
            
            if(loss_fnc == "ncc"):
                loss = self.ncc_loss_function(self.target_dict["image"], self.ground_truth["image"])
                
            elif(loss_fnc == "mi"):
                loss = self.mi_loss_function(self.target_dict["image"], self.ground_truth["image"])
            else:
                assert("Loss function must be either ncc or mi")
            #loss = bending_loss_fucntion(target_dict["image"])
            
            loss_log.append(loss.item())
            t.autograd.set_detect_anomaly(True)
            loss.backward(retain_graph=True)
            optimizer.step()
            print(f'Epoch: {epoch} Loss: {loss}')
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print (f'parameter {name} has value not equal to zero: {t.all(param.data == 0)}')
        #bring target into image shape
        self.target_dict["image"] = t.squeeze(self.target_dict["image"])
        self.target_dict= add_channel(self.target_dict)
        return self.target_dict, loss_log
    
    def ncc_loss_function(self, target_dict_image, ground_truth_image):
        monai_ncc = monai.losses.LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=5)
        loss = t.zeros(1)
        k = target_dict_image.shape[2]
        for sl in range(0,k):
            ##TTEST minus instead of  plus
            loss = loss + monai_ncc(target_dict_image[:,:,sl,:,:], ground_truth_image[:,:,sl,:,:])
        return loss

    def mi_loss_function(self, target_dict_image, ground_truth_image):
        monai_mi = monai.losses.GlobalMutualInformationLoss()
        loss = t.zeros(1)
        k = target_dict_image.shape[2]
        for sl in range(0,k):
            loss = loss + monai_mi(target_dict_image[:,:,sl,:,:], ground_truth_image[:,:,sl,:,:])
        return loss

    def bending_loss_fucntion(target_dict_image):
        monai_bending = monai.losses.BendingEnergyLoss()
        return monai_bending(target_dict_image.expand(-1,3,-1,-1,-1))