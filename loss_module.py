from multiprocessing import reduction
from turtle import forward
import monai 
import torch as t

class RegistrationLoss(t.nn.Module):
    def __init__(self,loss_fnc:str,device):
        super(RegistrationLoss,self).__init__()
        if loss_fnc == "ncc":
            self.monai_loss = monai.losses.LocalNormalizedCrossCorrelationLoss(spatial_dims=3, kernel_size=3)
        elif loss_fnc == "mi":
            self.monai_loss = monai.losses.GlobalMutualInformationLoss()
        else:
            assert("Please choose a valid loss function: either ncc or mi")
        self.device = device

    def forward(self, im_slices, fixed_image):
        """
        

        Parameters
        ----------
        im_slices : tensor
            slices from one stack
        fixed_image : dict
            current_fixed image

        Returns
        -------
        loss : TYPE
            DESCRIPTION.

        """
        n_slices = len(im_slices)

        fixed_image_image = fixed_image["image"].to(self.device)
        
        loss = t.zeros(1, requires_grad=True, device = self.device)
        
        for sl in range(0,n_slices):
            
            relevant_indices = t.nonzero(im_slices[sl,:,:,:,:], as_tuple = True)
            
            if len(relevant_indices[0]) > 0:
                min_ind = t.tensor([t.min(relevant_indices[i]).item() for i in range(len(relevant_indices))])
                max_ind = t.tensor([t.max(relevant_indices[i]).item() for i in range(len(relevant_indices))])
                
                pred = im_slices[sl, min_ind[0]:max_ind[0] + 1, min_ind[1]:max_ind[1] + 1, min_ind[2]:max_ind[2] + 1 ,min_ind[3]:max_ind[3] + 1].unsqueeze(0)
                target = fixed_image_image[0, min_ind[0]:max_ind[0] + 1, min_ind[1]:max_ind[1] + 1, min_ind[2]:max_ind[2] + 1 ,min_ind[3]:max_ind[3] + 1].unsqueeze(0)
                
                #print(f'pred: {str(pred.device)}, target: {str(target.device)}, loss: {str(loss.device)}')
                loss = loss + self.monai_loss(pred, target)
        return loss

class Loss_Volume_to_Volume(t.nn.Module):
    """
    class to calculate loss for initial 3d-3d registration
    """
    def __init__(self, loss_fnc:str, device) -> None:
        super().__init__()
        if loss_fnc == "ncc":
            self.monai_loss = monai.losses.LocalNormalizedCrossCorrelationLoss(spatial_dims=3, kernel_size=5)
        elif loss_fnc == "mi":
            self.monai_loss = monai.losses.GlobalMutualInformationLoss(reduction = "sum")
        else:
            assert("Please choose a valid loss function: either ncc or mi")
        self.device = device
    def forward(self, tr_fixed_tensor, stack_tensor):
        return self.monai_loss(tr_fixed_tensor,stack_tensor)

class Loss_Volume_to_Slice(t.nn.Module):
    def __init__(self,loss_fnc:str,device):
        super(Loss_Volume_to_Slice,self).__init__()
        if loss_fnc == "ncc":
            self.monai_loss = monai.losses.LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=7)
        elif loss_fnc == "mi":
            self.monai_loss = monai.losses.GlobalMutualInformationLoss(reduction = "sum")
        else:
            assert("Please choose a valid loss function: either ncc or mi")
        self.device = device

    def forward(self, tr_fixed_image, local_slices, n_slices, slice_dim):
        """
        Parameters
        ----------
        tr_fixed_image : tensor
            DESCRIPTION.
        local_slices : tensor
            DESCRIPTION.
        n_slices : int
            DESCRIPTION.
        slice_dim : int
            DESCRIPTION.

        Returns
        -------
        loss : tensor
            DESCRIPTION.

        """
        loss = t.zeros(1, device = self.device)
        for sl in range(0,n_slices):
            if slice_dim == 0:
                pred = tr_fixed_image[sl,:,sl,:,:]
                target = local_slices[sl,:,sl,:,:]
            elif slice_dim == 1:
                pred = tr_fixed_image[sl,:,:,sl,:]
                target = local_slices[sl,:,:,sl,:]
            elif slice_dim == 2:
                pred = tr_fixed_image[sl,:,:,:,sl]
                target = local_slices[sl,:,:,:,sl]
                #print(f'pred: {str(pred.device)}, target: {str(target.device)}, loss: {str(loss.device)}')
            loss = loss + self.monai_loss(pred.unsqueeze(0), target.unsqueeze(0))  
        return loss
    
