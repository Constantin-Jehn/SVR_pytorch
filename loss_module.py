from multiprocessing import reduction
from turtle import forward
import monai 
import torch as t

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
    def forward(self, tr_fixed_tensor:t.tensor, stack_tensor:t.tensor)->t.tensor:
        """
        return loss between the two tensors

        Args:
            tr_fixed_tensor (t.tensor):
            stack_tensor (t.tensor):

        Returns:
            t.tensor: loss tensor
        """
        return self.monai_loss(tr_fixed_tensor,stack_tensor)

class Loss_Volume_to_Slice(t.nn.Module):
    """class to calculate loss for 3d-2d registration
    """
    def __init__(self,loss_fnc:str,device):
        super(Loss_Volume_to_Slice,self).__init__()
        if loss_fnc == "ncc":
            self.monai_loss = monai.losses.LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=5)
        elif loss_fnc == "mi":
            self.monai_loss = monai.losses.GlobalMutualInformationLoss(reduction = "sum")
        else:
            assert("Please choose a valid loss function: either ncc or mi")
        self.device = device

    def forward(self, tr_fixed_tensor:t.tensor, local_slices:t.tensor, n_slices:int, slice_dim:int)->t.tensor:
        """_summary_

        Args:
            tr_fixed_tensor (t.tensor): transformed_tensor from fixed volume
            local_slices (t.tensor): slice as volume (ground truth)
            n_slices (int): number of slices in stack
            slice_dim (int): dimension along which stack is sliced

        Returns:
            t.tensor: loss tensor
        """
        loss = t.zeros(1, device = self.device)
        for sl in range(0,n_slices):
            if slice_dim == 0:
                pred = tr_fixed_tensor[sl,:,sl,:,:]
                target = local_slices[sl,:,sl,:,:]
            elif slice_dim == 1:
                pred = tr_fixed_tensor[sl,:,:,sl,:]
                target = local_slices[sl,:,:,sl,:]
            elif slice_dim == 2:
                pred = tr_fixed_tensor[sl,:,:,:,sl]
                target = local_slices[sl,:,:,:,sl]
                #print(f'pred: {str(pred.device)}, target: {str(target.device)}, loss: {str(loss.device)}')
            loss = loss + self.monai_loss(pred.unsqueeze(0), target.unsqueeze(0))
        return loss



        
    
