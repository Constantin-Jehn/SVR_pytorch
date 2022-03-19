"""
Module for the structure of volumes and 
variable name mainly follow Fast Volume Reconstruction From Motion Corrupted Stack of 2D slices (Kainz 2015)
"""

import numpy as np
import torch as t
import nibabel as nib
import time

class volume:
    
    def __init__(self,X=0,affine=t.eye(4)):
        self.X = X
        self.affine = affine
        self.p_r = 0
        self.p_s = 0
        self.geometry = 0
        self.n_voxels = 0
        
    # def __init__(self,geometry, n_voxels):
    #     self.p_r = self.create_voxel_mesh(geometry, n_voxels)
    #     self.geometry = geometry
    #     self.n_voxels = n_voxels
    #     self.X = 0
    #     self.update_X()
    #     self.affine = self.create_affine()
    
    def from_tensor(self, X, affine):
        self.X = X
        self.affine = affine
        
    def from_stack(self, geometry, n_voxels):
        """define volume starting from a stack"""
        self.p_r = self.create_voxel_mesh(geometry, n_voxels)
        self.geometry = geometry
        self.n_voxels = n_voxels
        self.X = 0
        self.update_X()
        self.affine = self.create_affine()
        
    # def create_pr_from_tensor(self):
        
        
    def create_voxel_mesh(self, geometry,n_voxels):
        """Function to create the voxel mesh. i.e. the target volume X R^(5,l*l)
        Inputs: 
        geometry: [[x_min, x_max],[y_min, y_max],[z_min, z_max]]
        voxels[n_x, n_y, n_z] number of voxels in each dimension
        """
        #lengths
        [l_x,l_y,l_z] = geometry[:,0] - geometry[:,1]
        x_lin, y_lin, z_lin = t.linspace(geometry[0,0],geometry[0,1],n_voxels[0]), t.linspace(geometry[1,0],geometry[1,1], n_voxels[1]), t.linspace(geometry[2,0],geometry[2,1], n_voxels[2])
        x_grid, y_grid, z_grid = t.meshgrid(x_lin, y_lin, z_lin)
        #coordinates
        coordinates = t.stack((t.flatten(x_grid),t.flatten(y_grid),t.flatten(z_grid)),dim = 0)
        #1 for "voxel structure" 0 for initial intensity value
        add_on = t.tensor([[1],[0]]).repeat(1,coordinates.shape[1])
        voxels = t.cat((coordinates,add_on),0)
        return voxels
    
    def p_s_tilde(self,F):
        """Function returns the "ideal" pixels in in the slice space R(4,n_voxels_total,n_slices_per_stack)"""
        #indexing on p_r because last row contains the intensity value
        indices =  t.matmul(F.float(),self.p_r[:4,:].float())
        p_s_tilde = t.cat((indices, self.p_r[4,:].unsqueeze(0)), dim=0)
        return p_s_tilde
    
    def create_ps_from_X(self):
        """create a pixel tensor from the volume X"""
        x_lin,y_lin = t.linspace(0,self.X.shape[0]-1,self.X.shape[0]), t.linspace(0,self.X.shape[1]-1,self.X.shape[1])
        x_grid, y_grid = t.meshgrid(x_lin, y_lin)
        coordinates = t.stack((t.flatten(x_grid), t.flatten(y_grid)), dim = 0)
        add_on = t.tensor([[0],[1],[0]]).repeat(1,coordinates.shape[1])
        pixels = t.cat((coordinates, add_on),0)
        pixels = pixels[:,:,None].repeat(1,1,self.X.shape[2])
        #add values for use during registration later
        for i in range(0,self.X.shape[2]):
            data = self.X[:,:,i]
            values = data.flatten()[None,:]
            pixels[4,:,i] = values
        self.p_s = pixels
        return self.p_s
    
    def update_X(self):
        X_shape  = (self.n_voxels[0].item(), self.n_voxels[1].item(), self.n_voxels[2].item())
        X = self.p_r[4,:].view(X_shape)
        self.X = X
    
    def create_affine(self):
        """creates the affine of a volume according to definition of geometry and resolution of voxels"""
        lengths = self.geometry[:,1] - self.geometry[:,0]
        zooms = t.diag(t.div(lengths,self.n_voxels))
        #translations = - self.geometry[:,0]
        translations = t.zeros(3)
        affine = t.tensor(nib.affines.from_matvec(zooms.numpy(),translations.numpy()))
        self.affine = affine
        return affine
     
    def reconstruct_stack(self, stack, time_it = False, batches = 1, dist_threshold = 1.5, sigma_PSN = 10):
        """Function to register 2d slice to volume"""
        sigmas = [sigma_PSN, sigma_PSN, sigma_PSN]
        for sl in range(0,stack.k):
            
            if time_it:
                t2 = time.time()
            
            #get F of current slice
            F = stack.F[:,:,sl]
            p_s_tilde = self.p_s_tilde(F)
            p_s = stack.p_s[:,:,sl]
            #transpose for distance tensor
            p_s_tilde_t_complete = p_s_tilde.transpose(0,1).float()
            p_s_t = p_s.transpose(0,1).float()
            
            if time_it:
                print(f'initialization: {time.time()-t2} s')
            
            #batchify not to run out of memory
            batch_size = round(p_s_tilde_t_complete.shape[0]/batches)
            lower_bound = 0
            upper_bound = batch_size
            for b in range(0,batches):
                #batch p_s_tilde
                if b < batches: 
                    p_s_tilde_t =p_s_tilde_t_complete[lower_bound:upper_bound,:]
                else:
                    p_s_tilde_t =p_s_tilde_t_complete[lower_bound:,:]
                
                if time_it:
                    t1 = time.time()
                #create distance tensor (n_voxels, n_pixels, 3) has difference in all three dimensions
                distance_tensor = t.abs(p_s_t[:,:3].unsqueeze(0) - p_s_tilde_t[:,:3].unsqueeze(1))
                
                if time_it:
                    print(f'distance vector: {time.time()-t1} s')
                    t2 = time.time()
                
                #for being inside voxel all distances (in voxel space) must be < 1
                #get indices that match from voxel to pixel [(voxel_index),(pixel_index)]
                indices = t.where(t.all(distance_tensor < dist_threshold,2))
                #indices = t.where(t.norm(distance_tensor,dim=2) < 1.5)
                #number of matches
                length = list(indices[0].shape)[0]
                
                if time_it:
                    print(f'distance evaluation: {time.time()-t2} s')
                    t2  = time.time()
                
                #extract relevant p_s and calculate PSN vectorized
                relevant_p_s = p_s[4,indices[1]]
                relevant_dist = distance_tensor[indices[0],indices[1],:]
                gauss = PSN_Gauss_vec(relevant_dist, sigmas=sigmas)
                value = t.multiply(gauss, relevant_p_s)
                
                #add values to corresponding voxels
                for voxel in range(0,length):
                    self.p_r[4,indices[0][voxel] + lower_bound] += value[voxel]
                
                lower_bound += batch_size
                upper_bound += batch_size
                
                if time_it:
                    print(f'voxel assignmet: {time.time()-t2} s')
            self.update_X()
            print(f'slice {sl} registered \n ')
        

def PSN_Gauss(offset,sigmas = [10,10,10]):
    """Gaussian pointspreadfunction higher sigma --> sharper kernel"""
    return t.exp(-(offset[0]**2)/sigmas[0]**2 - (offset[1]**2)/sigmas[1]**2 - (offset[2]**2)/sigmas[2]**2).float()
    
def PSN_Gauss_vec(offset, sigmas = [10,10,10]):
    """Vectorized Gaussian point spread function"""
    return t.exp(-t.div(offset[:,0]**2,sigmas[0]**2) -t.div(offset[:,1]**2,sigmas[2]**2) -t.div(offset[:,2]**2,sigmas[2]**2)).float()
         
    # def register_stack_euclid(self, stack):
    #     """Function to register a stack into the target volume"""
    #     #iterate over slices in stack
    #     for sl in range(0,stack.k):
    #         #tra
    #         F = stack.F[:,:,sl]
    #         p_s_tilde = self.p_s_tilde(F)
    #         p_s = stack.p_s[:,:,sl]
            
    #         p_s_tilde_t = p_s_tilde.transpose(0,1).float()
    #         p_s_t = p_s.transpose(0,1).float()
    #         distance_matrix = t.cdist(p_s_t[:,:4], p_s_tilde_t)
            
    #         #closest_ps_tilde = t.min(distance_matrix,1).indices
    #         #first index p_s_t, second tilde
    #         indices = t.where(distance_matrix < 2)
            
    #         distance_tensor = t.abs(p_s_t[indices[0],:3].unsqueeze(0) - p_s_tilde_t[indices[1],:3].unsqueeze(1))

    #         #extract relevant p_s and calculate PSN vectorized
    #         relevant_p_s = p_s[4,indices[1]]
    #         relevant_dist = distance_tensor[indices[0],indices[1],:]
    #         gauss = PSN_Gauss_vec(relevant_dist)
    #         value = t.multiply(gauss, relevant_p_s)
            
    #         #add values to corresponding voxels
    #         for voxel in range(0,length):
    #             self.p_r[4,indices[0][voxel]] += value[voxel]
            
    #         distance_vector = t.sub(p_s[:3,:],p_s_tilde[:3,closest_ps_tilde])
            
    #         for i in range(0,closest_ps_tilde.shape[0]):
    #             target_index = closest_ps_tilde[i]
    #             self.p_r[4,target_index] += PSN_Gauss(distance_vector[:,i])*p_s[4,i]
    #         self.update_X()
    #         print(f'slice {sl} registered')