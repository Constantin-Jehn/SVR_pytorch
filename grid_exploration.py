# %%
from torch.nn.functional import affine_grid
import monai
from monai.transforms import (
    AddChanneld,
    LoadImaged,
    ToTensord,
)
import torch as t
from copy import deepcopy
import torchio as tio
import numpy as np
import matplotlib.pyplot as plt

# %%
path = "sample_data/10_3T_nody_002.nii.gz"
add_channel = AddChanneld(keys=["image"])
loader = LoadImaged(keys=["image"])
to_tensor = ToTensord(keys=["image"])
monai_dict = {"image": path}
monai_image_layer = loader(monai_dict)
monai_image_layer = add_channel(add_channel(monai_image_layer))
monai_image_layer = to_tensor(monai_image_layer)

# %%
image_tensor = monai_image_layer["image"]
input_2d = image_tensor[:,:,100:200,100:200,12]
input_2d.shape

plt.figure()
plt.imshow(input_2d[0,0,:,:])
plt.title('input')
plt.show()

# %%
theta = t.eye(3)[:2,:].unsqueeze(0)
theta[0,0,2] = 0.0
size = input_2d.shape
align_corners = False
grid = affine_grid(theta,size, align_corners=align_corners)

# %%
output_2d = monai.networks.layers.grid_pull(input_2d,grid,interpolation = 'cubic', bound = "zero", extrapolate = True)

plt.figure()
plt.imshow(output_2d[0,0,:,:])
plt.title('transformed')
plt.show()


# %%
