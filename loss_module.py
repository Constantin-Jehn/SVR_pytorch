import monai 
import torch as t

class RegistrationLoss(t.nn.Module):
    def __init__(self,loss_fnc:str,device):
        super(RegistrationLoss,self).__init__()
        if loss_fnc == "ncc":
            self.monai_loss = monai.losses.LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=5)
        elif loss_fnc == "mi":
            self.monai_loss = monai.losses.GlobalMutualInformationLoss()
        else:
            assert("Please choose a valid loss function: either ncc or mi")
        self.device = device

    def forward(self, fixed_image, stack):
        stack_image, fixed_image_image = stack["image"], fixed_image["image"]
        n_slices = stack_image.shape[-1]
        loss = t.zeros(1, device = self.device)
        for sl in range(0,n_slices):
            loss = loss + self.monai_loss(stack_image[:,:,:,:,sl], fixed_image_image[:,:,:,:,sl])
            
        return loss
        