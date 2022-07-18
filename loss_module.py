import imp
from multiprocessing import reduction
from turtle import forward
import monai 
import torch as t

from voxelmorph_losses import ncc_loss


class Loss_Volume_to_Volume(t.nn.Module):
    """
    class to calculate loss for initial 3d-3d registration
    """
    def __init__(self, loss_fnc:str, device) -> None:
        super().__init__()
        if loss_fnc == "ncc":
            #self.monai_loss = monai.losses.LocalNormalizedCrossCorrelationLoss(spatial_dims=3, kernel_size=21)
            
            vol_vol_ncc_loss = monai.losses.LocalNormalizedCrossCorrelationLoss(spatial_dims=3, kernel_size=13)
            self.monai_loss = monai.losses.MaskedLoss(vol_vol_ncc_loss)
            
            
        elif loss_fnc == "mi":
            self.monai_loss = monai.losses.GlobalMutualInformationLoss(reduction = "sum")
        else:
            assert("Please choose a valid loss function: either ncc or mi")
        self.device = device
    def forward(self, tr_fixed_tensor:t.tensor, stack_tensor:t.tensor, mask:t.tensor)->t.tensor:
        """
        return loss between the two tensors
        Returns:
            t.tensor: loss tensor
        """
        mask = mask.to(self.device)
        return self.monai_loss(tr_fixed_tensor,stack_tensor,mask = mask)

class Loss_Volume_to_Slice(t.nn.Module):
    """class to calculate loss for 3d-2d registration
    """
    def __init__(self, kernel_size, loss_fnc:str,device):
        super(Loss_Volume_to_Slice,self).__init__()
        self.device = device
        self.kernel_size = kernel_size

        self.monai_ncc_loss = monai.losses.LocalNormalizedCrossCorrelationLoss(spatial_dims = 2, kernel_size = self.kernel_size)

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
            #loss = loss + ncc_loss(pred.unsqueeze(0),target.unsqueeze(0), device = self.device, win = (self.kernel_size, self.kernel_size))
            loss = loss + self.monai_ncc_loss(pred.unsqueeze(0),target.unsqueeze(0))
        return loss
 


class ncc(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, predicted:t.tensor, target:t.tensor) -> t.tensor:
        predicted,target = predicted.squeeze(),target.squeeze()

        std_pred, std_tgt = t.std(predicted), t.std(target)
        mu_pred, mu_tgt = t.mean(predicted), t.mean(target)
        n = t.numel(predicted)

        zero_mean_pred, zero_mean_tgt = predicted - mu_pred, target - mu_tgt

        nominator = t.sum((t.mul(zero_mean_pred,zero_mean_tgt))) + 1e-05
        denominator = (n * std_pred * std_tgt + 1e-05)

        return t.div(nominator,denominator)




