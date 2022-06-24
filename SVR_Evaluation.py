from numpy import float128
import monai
import torchio as tio
import torch as t
from SVR_Preprocessor import Preprocesser
import utils


def psnr(fixed_image:dict, stacks:list, n_slices:int, tio_mode:str)->float128:
    preprocessor = Preprocesser('','','',[],'','cpu','','welch')
    fixed_tio = utils.monai_to_torchio(fixed_image)

    n_slices_total = 0
    psnr = 0
    for st in range(0,len(stacks)):
        n_slices_total +=n_slices[st]
        stack_tio = utils.monai_to_torchio(stacks[st])
        resampler = tio.Resample(stack_tio, image_interpolation=tio_mode)
        fixed_resampled = resampler(fixed_tio)
        max_val, min_val = t.max(t.max(fixed_resampled.tensor),t.max(stack_tio.tensor)), t.min(t.min(fixed_resampled.tensor),t.min(stack_tio.tensor))
        max_val = max_val - min_val
        psnr_metric = monai.metrics.PSNRMetric(max_val = max_val, reduction = 'mean')

        for sl in range(0,n_slices[st]):
            pred, target = fixed_resampled.tensor[0,:,:,sl], stack_tio.tensor[0,:,:,sl]
            psnr+=psnr_metric(pred,target)
    
    return t.mean(t.div(psnr,n_slices_total))





