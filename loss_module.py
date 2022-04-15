import monai 
import torch as t

class RegistrationLoss(t.nn.Module):
    def __init__(self,loss_fnc:str,device):
        super(RegistrationLoss,self).__init__()
        if loss_fnc == "ncc":
            self.monai_loss = monai.losses.LocalNormalizedCrossCorrelationLoss(spatial_dims=1, kernel_size=1)
        elif loss_fnc == "mi":
            self.monai_loss = monai.losses.GlobalMutualInformationLoss()
        else:
            assert("Please choose a valid loss function: either ncc or mi")
        self.device = device

    def forward(self, im_slices, fixed_image):
        n_slices = len(im_slices)

        fixed_image_image = fixed_image["image"]
        
        
        loss = t.zeros(1, device = self.device)
        
        for sl in range(0,n_slices):
            
            relevant_indices = t.nonzero(im_slices[sl], as_tuple = True)
            
            pred, target = im_slices[sl][relevant_indices].unsqueeze(0).unsqueeze(0), fixed_image_image[relevant_indices].unsqueeze(0).unsqueeze(0)
            
            loss = loss + self.monai_loss(pred, target)
            
        return loss
        