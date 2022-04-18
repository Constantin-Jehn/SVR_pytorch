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
        n_slices = len(im_slices)

        fixed_image_image = fixed_image["image"].to(self.device)
        
        loss = t.zeros(1, device = self.device)
        
        for sl in range(0,n_slices):
            
            relevant_indices = t.nonzero(im_slices[sl,:,:,:,:], as_tuple = True)
            
            min_ind = t.tensor([t.min(relevant_indices[i]).item() for i in range(len(relevant_indices))])
            max_ind = t.tensor([t.max(relevant_indices[i]).item() for i in range(len(relevant_indices))])
            
            pred = im_slices[sl, min_ind[0]:max_ind[0] + 1, min_ind[1]:max_ind[1] + 1, min_ind[2]:max_ind[2] + 1 ,min_ind[3]:max_ind[3] + 1].unsqueeze(0)
            target = fixed_image_image[0, min_ind[0]:max_ind[0] + 1, min_ind[1]:max_ind[1] + 1, min_ind[2]:max_ind[2] + 1 ,min_ind[3]:max_ind[3] + 1].unsqueeze(0)
            
            #print(f'pred: {str(pred.device)}, target: {str(target.device)}, loss: {str(loss.device)}')
            loss = loss + self.monai_loss(pred, target)
            
        return loss
        