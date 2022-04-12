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
        self.k = len(self.stack_filenames)
        self.mask_filename = mask_filename
        self.pixdim = pixdim
        self.device = device
        self.mode = mode
        
        self.crop_images()

        self.ground_truth = self.load_stacks()
        #self.resample_to_pixdim()
        self.stacks = self.load_stacks()
        
        
        self.resample_to_hr()
        self.resample_to_common_coord()
        self.fixed_image = self.create_common_volume()
        add_channel = AddChanneld(keys=["image"])
        self.fixed_image = add_channel(self.fixed_image)
        
        self.stacks = self.load_stacks()
        
        
        #slices is a list: each entry is a list of all slices of one stack
        self.slices, self.n_slices_max = self.construct_slices()
        print("preprocessing done")
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
        
        for i in range(0,self.k):
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
    
    def resample_to_pixdim(self):
        """
        After cropping and having saved the ground truth image, bring the images
        to desired resolution for reconstruction

        Returns
        -------
        None.
        """
        resampler = tio.transforms.Resample(self.pixdim)
        
        for i in range(0,self.k):
            filename = self.stack_filenames[i]
            path_stack = os.path.join(self.prep_folder, filename)
            stack = tio.ScalarImage(path_stack)
            resampled_stack = resampler(stack)
            path_dst = os.path.join(self.prep_folder, filename)
            resampled_stack.save(path_dst)    
    
    
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
        
        stack_list = list()
        for i in range(0,self.k):
            path = os.path.join(self.prep_folder, self.stack_filenames[i])
            stack_dict = {"image": path}
            stack_dict = loader(stack_dict)
            stack_dict = to_tensor(stack_dict)
            stack_dict = add_channel(stack_dict)
            #keep meta data correct
            stack_dict["image_meta_dict"]["spatial_shape"] = np.array(list(stack_dict["image"].shape)[1:])
            stack_list.append(stack_dict)
        return stack_list
            
    def construct_slices(self):
        """
        from the prepocessed (cropped + corr. resolution) image slices are 
        created that will be used to perform the rotations/translations on
        Returns
        -------
        slices : list
            each entry belongs to one image, and contains again a list of all 
            the slices represented as 3d torch tensor

        """
        #slices are used in the reconstruction model where formate (batch,channel, HWD) is necessary
        add_channel = AddChanneld(keys=["image"])
        slices = list()
        
        n_slices_max = 0 
        for st in range(0,self.k):
            stack = add_channel(self.stacks[st])
            stack_image = stack["image"]
            n_slices = stack_image.shape[-1]
            if n_slices > n_slices_max: 
                n_slices_max = n_slices
            im_slices = list()
            for i in range (0,n_slices):
                tmp = deepcopy(stack_image)
                tmp[:,:,:,:,:i] = 0
                tmp[:,:,:,:,i+1:] = 0
                im_slices.append(tmp)
            slices.append(im_slices)
        return slices, n_slices_max
    
    
    def update_local_stack(self, im_slices, n_stack):
        """
        use to update the stacks after transform (which should be optimized)
        was applied to slices
        Parameters
        ----------
        im_slices : list 
            contains slices of one stack
        n_stack : int
            index of stack to be updated
        Returns
        -------
        None
        """
        n_slices = len(im_slices)
        tmp = t.zeros(im_slices[0].shape)
        for sli  in range(0,n_slices):
            tmp = tmp + im_slices[sli]
        #update target_dict
        self.stacks[n_stack]["image"] = tmp
    
    
    def batch_to_image(self, stacks):
        """
        casts stacks from batch format (batch,channel,H,W,D)
        to image format(channel,H,W,D)

        Parameters
        ----------
        stacks : list
            list of stacks

        Returns
        -------
        stacks : TYPE
            DESCRIPTION.

        """
        
        for i in range(0,self.k):
            stacks[i]["image"] = t.squeeze(stacks[i]["image"]).unsqueeze(0)
        return stacks
    
    
    def resample_to_hr(self, n_stack:int = -1):
        """
        Resamples stacks to high resolution

        Parameters
        ----------
        n_stack : TYPE, optional, int
           if it is set > 0 only the indexed stack is updated

        Returns
        -------
        None.

        """
        
        spacing_d = monai.transforms.Spacing(pixdim = self.pixdim, 
                                            padding_mode= "zeros")
        if n_stack < 0: 
            for st in range(0,self.k):
                self.stacks[st]["image"], self.stacks[st]["image_meta_dict"]["original_affine"],  self.stacks[st]["image_meta_dict"]["affine"] =  spacing_d(self.stacks[n_stack]["image"], affine = self.stacks[st]["image_meta_dict"]["affine"])
                #keep importan meta data up-todate
                self.stacks[st]["image_meta_dict"]["pixdim"][1:4] = np.array(self.pixdim)
                self.stacks[st]["image_meta_dict"]["spatial_shape"] = np.array(list(self.stacks[st]["image"].shape[1:]))
        else:
            self.stacks[n_stack]["image"], self.stacks[n_stack]["image_meta_dict"]["original_affine"],  self.stacks[n_stack]["image_meta_dict"]["affine"] =  spacing_d(self.stacks[n_stack]["image"], affine = self.stacks[n_stack]["image_meta_dict"]["affine"])
            #keep importan meta data up-todate
            self.stacks[n_stack]["image_meta_dict"]["pixdim"][1:4] = np.array(self.pixdim)
            self.stacks[n_stack]["image_meta_dict"]["spatial_shape"] = np.array(list(self.stacks[n_stack]["image"].shape[1:]))
                
        

    def resample_to_common_coord(self, n_stack = - 1, to_fixed = False):
        """
        brings all stacks to common coordinate system
        Parameters
        ----------
        resamplers : list
            resampler objects one for each stack
        n_stack: int
            if n_stack > 0 only one index is resa

        Returns
        -------
        None.

        """
        if to_fixed:
            dst_meta = self.fixed_image["image_meta_dict"]
        else:
            dst_meta = self.stacks[0]["image_meta_dict"]
        resampler = monai.transforms.ResampleToMatch()
        
        if n_stack < 0 :
            for st in range(1,self.k):
                file_obj = deepcopy(self.stacks[st]["image_meta_dict"]["filename_or_obj"])
                original_affine = deepcopy(self.stacks[st]["image_meta_dict"]["original_affine"])
                self.stacks[st]["image"], self.stacks[st]["image_meta_dict"] = resampler(self.stacks[st]["image"],src_meta = self.stacks[st]["image_meta_dict"], 
                                                                                              dst_meta = dst_meta, padding_mode = "zeros")
                
                self.stacks[st]["image_meta_dict"]["filename_or_obj"] = file_obj
                self.stacks[st]["image_meta_dict"]["original_affine"] = original_affine
        else:
             file_obj = deepcopy(self.stacks[n_stack]["image_meta_dict"]["filename_or_obj"])
             original_affine = deepcopy(self.stacks[n_stack]["image_meta_dict"]["original_affine"])
             self.stacks[n_stack]["image"], self.stacks[n_stack]["image_meta_dict"] = resampler(self.stacks[n_stack]["image"],src_meta = self.stacks[n_stack]["image_meta_dict"], 
                                                                                           dst_meta = dst_meta, padding_mode = "zeros")
             
             self.stacks[n_stack]["image_meta_dict"]["filename_or_obj"] = file_obj
             self.stacks[n_stack]["image_meta_dict"]["original_affine"] = original_affine       
     
        
    def create_common_volume(self):
        """
        Combine updated local stacks that are in common coordinate system, to 
        one superpositioned volume
        


        Returns
        -------
        world_stack : dictionary
            contains nifti file of the the reconstructed brain.

        """
        
        tmp = t.zeros_like(self.stacks[0]["image"])
        for st in range(1,self.k):
            tmp = tmp + self.stacks[st]["image"]
        world_stack = {"image":tmp, "image_meta_dict": self.stacks[0]["image_meta_dict"]}
        return world_stack
    
    
    def resample_world_to_ground_coords(self, world_stack):
        resampler = monai.transforms.ResampleToMatch()
        for st in range(0,self.k):
            self.stacks[st]["image"], self.stacks[st]["image_meta_dict"] = resampler(world_stack["image"],src_meta = world_stack["image_meta_dict"], 
                                                                                          dst_meta = self.ground_truth[st]["image_meta_dict"],
                                                                                          padding_mode = "zeros")
        
    def ncc_loss_function(self, n_stack = -1):
        """
        Calculate ncc loss as sum from all stacks or only from a single stack.
        Stacks are here resamples into their initial cooridinates and resolution.
        The image data is from the reconstructed volume in common coordinates
        Parameters:
        -------
        n_stack: int
            if > -1: loss is only calculated for one the stack of that index
        
        Returns
        -------
        loss : t.tensor
            the loss
            

        """
        
        monai_ncc = monai.losses.LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=5)
        
        stack = self.stacks[n_stack]
        #onl resample image keep local meta_data in-tact
        stack_image, fixed_image_image = stack["image"], self.fixed_image["image"]
        n_slices = stack_image.shape[-1]
        
        loss = t.zeros(1)
        for sl in range(0,n_slices):
            loss = loss + monai_ncc(stack_image[:,:,:,:,sl], fixed_image_image[:,:,:,:,sl])
        return loss
    
    
    def optimize_multiple_stacks(self, epochs, lr, loss_fnc = "ncc", opt_alg = "Adam"):
        add_channel = AddChanneld(keys=["image"])
    
        n_stacks = len(self.stacks)
        # resamplers = list()
        # #create resampler for each stack to make use inverse function later on
        # for i in range(0,n_stacks):
        #     resamplers.append(monai.transforms.ResampleToMatch())
        
        models = list()
        for st in range (0,self.k):
            n_slices = len(self.slices[st])
            models.append(reconstruction_model.Reconstruction(n_slices = n_slices, device = self.device))

        loss_log = list()
        
        for st in range(0,self.k):
            print(f"\n  stack: {st}")
            model = models[st]
            slices_tmp = self.slices[st]
            
            if opt_alg == "SGD":
                optimizer = t.optim.SGD(model.parameters(), lr = lr)
            elif(opt_alg == "Adam"):
                optimizer = t.optim.Adam(model.parameters(), lr = lr)
            else:
                assert("Choose SGD or Adam as optimizer")
                  
            for epoch in range(0,epochs):
                model.train()
                optimizer.zero_grad()
            
                transformed_slices = model(slices_tmp)
                
                self.stacks[st] = add_channel(self.stacks[st])
                
                #only changes image
                self.update_local_stack(transformed_slices, n_stack = st)
                
                #batch to image
                self.stacks[st]["image"] = t.squeeze(self.stacks[st]["image"]).unsqueeze(0)
                
                #self.resample_to_hr(n_stack = st)
                
                self.resample_to_common_coord(n_stack=st, to_fixed=True)
                
                self.stacks[st] = add_channel(self.stacks[st])
                
                
                
                #self.resample_to_common_coord(n_stack = st)
                
                loss = self.ncc_loss_function(n_stack = st)
                #here stack[st] is in coordinates of fixed image
                print(f'Epoch: {epoch} loss: {loss.item()}')
                
                loss.backward(retain_graph=True)
                
                optimizer.step()
            # get last update 
            #self.update_local_stack(transformed_slices, n_stack = st)
            self.stacks[st]["image"] = t.squeeze(self.stacks[st]["image"]).unsqueeze(0)
            
        #self.resample_to_common_coord(to_fixed=True)    
        world_stack = self.create_common_volume()
              
        return world_stack
            #todo: loss for multiple

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
            self.target_dict = model(self.im_slices, self.target_dict)
            
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
    
    def ncc_loss_function_single_stack(self, stack_image, ground_truth_image):
        monai_ncc = monai.losses.LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=5)
        loss = t.zeros(1)
        k = stack_image.shape[-1]
        for sl in range(0,k):
            ##TTEST minus instead of  plus
            loss = loss + monai_ncc(stack_image[:,:,:,:,sl], ground_truth_image[:,:,:,:,sl])
        return loss

    def mi_loss_function_single_stack(self, target_dict_image, ground_truth_image):
        monai_mi = monai.losses.GlobalMutualInformationLoss()
        loss = t.zeros(1)
        k = target_dict_image.shape[2]
        for sl in range(0,k):
            loss = loss + monai_mi(target_dict_image[:,:,:,:,sl], ground_truth_image[:,:,:,:,sl])
        return loss

    def bending_loss_fucntion_single_stack(target_dict_image):
        monai_bending = monai.losses.BendingEnergyLoss()
        return monai_bending(target_dict_image.expand(-1,3,-1,-1,-1))