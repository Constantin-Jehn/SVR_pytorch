import torch as t
import volume
import stack
import utils


class Reconstruction(t.nn.Module):
    def __init__(self, target_volume:volume.volume, rec_stack:stack.stack):
        super().__init__()
        self.k = rec_stack.k
        self.rec_stack = rec_stack
        self.target_volume = target_volume
        self.rotations = t.nn.Parameter(t.zeros(3,self.k))
        self.translations = t.nn.Parameter(t.zeros(3,self.k))
        
    def forward(self):
        #update Transition matrices of the stack
        for sl in range(0,self.k):
            T = utils.create_T(self.rotations[:,sl], self.translations[:,sl])
            self.rec_stack.T[:,:,sl] = T
        self.rec_stack.create_F()
        self.rec_stack.create_F_inv()
        self.target_volume.reconstruct_stack(self.rec_stack, batches = 5)
        
        return self.target_volume

