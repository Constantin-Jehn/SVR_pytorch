from turtle import forward
from numpy import float128
import monai
import torchio as tio
import torch as t
from SVR_Preprocessor import Preprocesser
import utils


def psnr(fixed_image:dict, stacks:list, n_slices:int, tio_mode:str, resampled_masks:list)->float128:
    """
    """

    fixed_tio = utils.monai_to_torchio(fixed_image)

    n_slices_total = 0
    psnr = 0
    #pictures are normalized between zero and 1
    max_value = 1
    psnr_metric = PSNR(device = fixed_tio.data.device, max_value=max_value)
    #psnr_metric = monai.metrics.PSNRMetric(max_val=1,reduction = "mean")
    n_slices_total = 0
    for st in range(0,len(stacks)):

        
        stack_tio = utils.monai_to_torchio(stacks[st])
        resampler = tio.Resample(stack_tio, image_interpolation=tio_mode)
        fixed_resampled = resampler(fixed_tio)

        fixed_resampled.set_data(t.nan_to_num(fixed_resampled.tensor, nan = 0))

        mask = resampled_masks[st].tensor.to(stack_tio.tensor.device)
        #psnr_metric = monai.metrics.PSNRMetric(max_val = 1, reduction = 'mean')
        

        for sl in range(0,n_slices[st]):
            mask_slice = mask[0,:,:,sl]
            #change 0 to 2 in the mask, images are normalized betw 0 and 1 --> 
            pred, target = fixed_resampled.tensor[0,:,:,sl], utils.normalize_zero_to_one(stack_tio.tensor[0,:,:,sl])

            #take only parts of tensor inside mask --> output is 1D
            pred_masked, target_masked = pred[mask_slice == 1], target[mask_slice == 1]
            
            #unsqueeze bec. monai psnr needs a batch first
            #1dim vector is ok, psnr doesn't take spatial closeness into account
            if t.numel(pred_masked) > 0:
                psnr_tmp = t.mean(psnr_metric(pred_masked.unsqueeze(0),target_masked.unsqueeze(0)))
                n_slices_total = n_slices_total + 1
            """
            finite_entries =  t.isfinite(psnr_raw_res)
            psnr_clean = t.sum(psnr_raw_res[finite_entries]) / t.sum(finite_entries)
            """
            psnr+=psnr_tmp
    
    return t.mean(t.div(psnr,n_slices_total))


class PSNR(t.nn.Module):
    def __init__(self, device, max_value = 1) -> None:
        super().__init__()
        self.max = t.tensor(max_value, device=device)
        self.mse_loss = t.nn.MSELoss(reduction='mean')
    
    def forward(self, pred:t.tensor, target:t.tensor):
        """
        Calculated PSNR between two images pred and target

        Args:
            pred (t.tensor): predicted image
            target (t.tensor): target image
        """
        #add tiny number for numerical stability
        mse = self.mse_loss(pred,target) + 1e-5

        return 20 * t.log10(self.max) - 10 * t.log10(mse)





