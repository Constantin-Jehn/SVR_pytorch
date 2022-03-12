import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import open3d as o3d

import torch as t

import transformations as trans

class stack:
    """models one stack of images"""
    def __init__(self, I, affine, beta):
        #stacked slices R^(l,l,k)
        self.I = I
        self.k = self.I.shape[2]
        #affine transformation to RAS system == W_r
        self.affine = affine.float()
        #data matrix R^(l*l,k)
        self.A = t.flatten(I,0,1)
        #error matrix
        self.E = t.zeros_like(self.A)
        #observation matrix
        self.D = self.A.clone().detach()
        #hyperparemter as threshold for calculating D_prime
        self.beta = beta
        #list of unknowm rigid body transformations initalize with identites R^(4,4,k)
        self.T = t.eye(4).unsqueeze(2).repeat(1,1,self.k)
        #add initial translation in z-direction
        self.T[2,3,:] = t.linspace(0,self.k-1,self.k)
        #list ofimage to world transformations
        self.W_s = t.eye(4).unsqueeze(2).repeat(1,1,self.k)
        self.F = self.F()
        self.F_inv = self.F_inv()
        #matrix of corresponding voxel R(4,l*l*)
        self.p_r = self.p_r
    
    def rank_D(self):
        return t.matrix_rank(self.D).item()
    
    def within_stack_error(self):
        """Funktion for surrogate measure of omega according to Eq (3) in Kainz 2015"""
        U,S,V = t.svd(self.D)
        S = t.diag(S)
        error = self.beta + 1
        r=0
        while error > self.beta:
            U_prime = U[:,:r]
            S_prime = S[:r,:r]
            V_prime_t = V[:,:r].transpose(0,1)
            D_prime = t.matmul(t.matmul(U_prime, S_prime),V_prime_t)
            #error measure in equation (2)
            error = t.norm(self.D-D_prime, p='fro') / t.norm(self.D, p='fro')
            r +=1
        return (r*error).item()
    
    def F(self):
        """Function to calculate F, mapping pixels from the stack to voxels world coordinates R^(4,4,k)"""
        #inverse is batch_first --> therefore permute
        W_s_inv = t.linalg.inv(self.W_s.permute(2,0,1).float())
        W_s_inv = W_s_inv.permute(1,2,0)
        T_inv = t.linalg.inv(self.T.permute(2,0,1).float())
        T_inv = T_inv.permute(1,2,0)
        F = t.einsum('ijb,jnb->inb',W_s_inv,T_inv)
        F = t.einsum('ijb,jn->inb',F,self.affine.float())
        self.F = F
        return F.float()
    
    def F_inv(self):
        """Function to calculate F_invers, voxels from world coordinates to pixels in stack R^(4,4,k)"""
        W_inv = t.linalg.inv(self.affine.float()).float()
        F_inv = t.einsum('ij,jnb->inb',W_inv.float(),self.T.float())
        F_inv = t.einsum('ijb,jnb->inb',F_inv.float(),self.W_s.float())
        self.F_inv = F_inv
        return F_inv.float()
    
    def p_r(self):
        """Function to calculate p_r the continuous voxel position in world coordinates"""
        x_lin,y_lin,k_lin = t.linspace(0,self.I.shape[0]-1,self.I.shape[0]), t.linspace(0,self.I.shape[1]-1,self.I.shape[1]), t.linspace(0,self.I.shape[2]-1,self.I.shape[2])
        x_grid, y_grid = t.meshgrid(x_lin, y_lin)
        coordinates = t.stack((t.flatten(x_grid), t.flatten(y_grid)), dim = 0)
        add_on = t.tensor([[0],[1]]).repeat(1,coordinates.shape[1])
        pixels = t.cat((coordinates, add_on),0)
        output = t.einsum('ijk,jl->ilk',self.F,pixels)
        #switch order for consistency with 2d coordinates
        k_grid, x_grid, y_grid = t.meshgrid(k_lin, x_lin, y_lin)
        coordinates_3d = t.stack((t.flatten(x_grid), t.flatten(y_grid), t.flatten(k_grid)), dim = 0).int().numpy()
        values = self.I[coordinates_3d][None,:]
        output = output.reshape(output.shape[0],output.shape[1]*output.shape[2])
        output = t.cat((output,values),0)
        self.p_r = output
        return self.p_r

    

def PSN_Gauss(offset,sigmas = [1,1,1]):
    """Gaussian pointspreadfunction higher sigma --> sharper kernel"""
    return float(np.exp(-(offset[0]**2)/sigmas[0]**2 - (offset[1]**2)/sigmas[1]**2 - (offset[2]**2)/sigmas[2]**2))
    

class volume:
    def __init__(self,geometry, n_voxels):
        self.X = self.create_voxel_mesh(geometry, n_voxels)
        
    def create_voxel_mesh(geometry,n_voxels):
        """Function to create the voxel mesh. i.e. the target volume X
        Inputs: 
        geometry: [[x_min, x_max],[y_min, y_max],[z_min, z_max]]
        voxels[n_x, n_y, n_z] numer of voxels in each dimension
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
        
    
    
        
    