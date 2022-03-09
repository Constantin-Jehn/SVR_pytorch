import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import torch as t

import transformations as trans

class stack:
    def __init__(self, I, affine, beta):
        #stacked slices R^(l,l,k)
        self.I = I
        self.k = self.I.shape[2]
        #affine transformation to RAS system == W_r
        self.affine = affine
        #data matrix R^(l*l,k)
        self.A = t.flatten(I,0,1)
        #error matrix
        self.E = t.zeros_like(self.A)
        #observation matrix
        self.D = self.A.clone().detach()
        #hyperparemter as threshold for calculating D_prime
        self.beta = beta
        #list of unknowm rigid body transformations initalize with identites
        self.T = t.eye(4).unsqueeze(2).repeat(1,1,self.k)
        #list ofimage to world transformations
        self.W_s = t.eye(4).unsqueeze(2).repeat(1,1,self.k)
    
    def rank_D(self):
        return t.matrix_rank(self.D).item()
    
    def within_stack_error(self):
        U,S,V = t.svd(self.D)
        S = t.diag(S)
        error = self.beta + 1
        r=0
        while error > self.beta:
            U_prime = U[:,:r]
            S_prime = S[:r,:r]
            V_prime_t = V[:,:r].transpose(0,1)
            D_prime = t.matmul(t.matmul(U_prime, S_prime),V_prime_t)
            error = t.norm(self.D-D_prime, p='fro') / t.norm(self.D)
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
        return F
    
    def F_inv(self):
        """Function to calculate F_invers, voxels from world coordinates to pixels in stack R^(4,4,k)"""
        W_inv = t.linalg.inv(self.affine).float()
        F_inv = t.einsum('ijb,jnb->inb',self.T,self.W_s)
        F_inv = t.einsum('ij,jnb->inb',W_inv,F_inv)
        return F_inv