import imp
from multiprocessing import reduction
from turtle import forward
import monai
from numpy import dtype 
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
    def __init__(self, kernel_size, loss_fnc:str,device, lambda_reg:float, sigma:float):
        super(Loss_Volume_to_Slice,self).__init__()
        self.device = device
        self.kernel_size = kernel_size
        self.lambda_reg = lambda_reg
        self.sigma = sigma

        #self.monai_ncc_loss = monai.losses.LocalNormalizedCrossCorrelationLoss(spatial_dims = 2, kernel_size = self.kernel_size)
        vol_slice_monai_ncc_loss = monai.losses.LocalNormalizedCrossCorrelationLoss(spatial_dims = 2, kernel_size = self.kernel_size)
        self.monai_ncc_loss = monai.losses.MaskedLoss(vol_slice_monai_ncc_loss)

    def forward(self, tr_fixed_tensor:t.tensor, local_slices:t.tensor, n_slices:int, slice_dim:int, resampled_mask:t.tensor)->t.tensor:
        """_summary_

        Args:
            tr_fixed_tensor (t.tensor): transformed_tensor from fixed volume
            local_slices (t.tensor): slice as volume (ground truth)
            n_slices (int): number of slices in stack
            slice_dim (int): dimension along which stack is sliced
            resampled_mask(t.tensor): segmentation mask resampled to current stack

        Returns:
            t.tensor: loss tensor
        """
        loss = t.zeros(1, device = self.device)
        resampled_mask_tensor = resampled_mask.tensor.to(self.device)
        #reg = edge_preserving_regularizer(tr_fixed_tensor[0,0,:,:,:], resampled_mask, sigma = self.sigma)
        for sl in range(0,n_slices):
            if slice_dim == 0:
                pred = tr_fixed_tensor[sl,:,sl,:,:]
                target = local_slices[sl,:,sl,:,:]
                mask = resampled_mask_tensor[:,sl,:,:]
            elif slice_dim == 1:
                pred = tr_fixed_tensor[sl,:,:,sl,:]
                target = local_slices[sl,:,:,sl,:]
                mask = resampled_mask_tensor[:,:,sl,:]
            elif slice_dim == 2:
                pred = tr_fixed_tensor[sl,:,:,:,sl]
                target = local_slices[sl,:,:,:,sl]
                mask = resampled_mask_tensor[:,:,:,sl]
                #print(f'pred: {str(pred.device)}, target: {str(target.device)}, loss: {str(loss.device)}')
            #loss = loss + ncc_loss(pred.unsqueeze(0),target.unsqueeze(0), device = self.device, win = (self.kernel_size, self.kernel_size))
            loss = loss + self.monai_ncc_loss(pred.unsqueeze(0),target.unsqueeze(0), mask = mask.unsqueeze(0))
        return loss 
 
def edge_preserving_regularizer(X:t.tensor, mask:t.tensor, sigma:float)->float:
    """

    Args:
        X (t.tensor): fixed_image, resampled (H,W,D)
        mask (t.tensor): segementation mask (b,H,W,D)
        sigma (float): _description_

    Returns:
        float: value of regularization term
    """
    #indices inside mask
    regularizer = t.zeros(1, device=X.device, requires_grad=True)
    #gives indices as tuple (tensor(x_indices),tensor(y_indices),tensor(z_indices))
    relevant_indices = (mask.tensor).nonzero(as_tuple = True)[-3:]
    for voxel in range(0,len(relevant_indices[0])):
        #get index in X
        i = (relevant_indices[0][voxel],relevant_indices[1][voxel],relevant_indices[0][voxel])
        neighbour_indices, d = neighbour_voxels(i,X.shape[-3:], X.device)
        X_center_value = X[i]
        for n in range(0,len(neighbour_indices)):
            argument = t.div( (X[neighbour_indices[n]] - X_center_value), sigma * d[n])
            regularizer = regularizer + phi(argument)
    return regularizer

def neighbour_voxels(i:tuple, X_shape:tuple, X_device:str)->tuple:
    """Calculates neighbouring indices and gives back those indices and the norm of the vectors d between i and the neighbours

    Args:
        i (tuple): central index
        X_shape (tuple): shape of common volume

    Returns:
        tuple: neighbour_indices(list), d(t.tensor)
    """
    neighbour_indices = list()
    for x_val in [-1,0,1]:
        x_coord = i[0] + x_val
        for y_val in [-1,0,1]:
            y_coord = i[1] + y_val
            for z_val in [-1,0,1]:
                 z_coord = i[2] + z_val
                 inside = x_coord in range(0,X_shape[0]) and y_coord in range(0,X_shape[1]) and z_coord in range(X_shape[2])
                 not_center = not(x_val == 0 and y_val == 0 and z_val==0)
                 if inside and not_center:
                     neighbour_indices.append((x_coord,y_coord,z_coord))
    vecs = t.tensor(neighbour_indices,device=X_device,dtype = float) - t.tensor(i,device=X_device,dtype = float)
    d = t.norm(vecs, dim = 1)
    return neighbour_indices, d


def phi(t_var:float) -> float:
    """formula phi(t) from Kuklisove (2012)

    Args:
        t (float): see equation (3)

    Returns:
        float: output of simple function
    """
    return 2 * t.sqrt(1 + t.pow(t_var,t.tensor(2,device = t_var.device))) - 2


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




