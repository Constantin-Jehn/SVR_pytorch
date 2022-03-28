"""
Module for the structure of stacks and 
variable name mainly follow Fast Volume Reconstruction From Motion Corrupted Stack of 2D slices (Kainz 2015)
"""
import torch as t
import utils
class stack:
    """models one stack of images"""
    def __init__(self, I, affine, add_init_offset = True, beta=0.1):
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
        if add_init_offset:
            self.T[2,3,:] = t.linspace(0,self.k-1,self.k) 
        #list ofimage to world transformations
        self.W_s = t.eye(4).unsqueeze(2).repeat(1,1,self.k)
        self.F = self.create_F()
        self.F_inv = self.create_F_inv()
        #pixel representation R(5,l*l) (i,j,0,1,value)
        self.p_s = self.create_p_s()
        #matrix of corresponding voxel R(5,l*l,k) (x,y,z,0,value)
        self.p_r = self.create_p_r()
    
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
    
    def create_F(self):
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
    
    def create_F_inv(self):
        """Function to calculate F_invers, voxels from world coordinates to pixels in stack R^(4,4,k)"""
        W_inv = t.linalg.inv(self.affine.float()).float()
        F_inv = t.einsum('ij,jnb->inb',W_inv.float(),self.T.float())
        F_inv = t.einsum('ijb,jnb->inb',F_inv.float(),self.W_s.float())
        self.F_inv = F_inv
        return F_inv.float()
    
    def create_p_s(self):
        """Function to create represenation of image in voxel space R^(5,l*l,k) (i,j,0,1,value)"""
        x_lin,y_lin = t.linspace(0,self.I.shape[0]-1,self.I.shape[0]), t.linspace(0,self.I.shape[1]-1,self.I.shape[1])
        x_grid, y_grid = t.meshgrid(x_lin, y_lin, indexing='ij')
        coordinates = t.stack((t.flatten(x_grid), t.flatten(y_grid)), dim = 0)
        add_on = t.tensor([[0],[1],[0]]).repeat(1,coordinates.shape[1])
        pixels = t.cat((coordinates, add_on),0)
        pixels = pixels[:,:,None].repeat(1,1,self.k)
        #add values for use during registration later
        for i in range(0,self.k):
            data = self.I[:,:,i]
            values = data.flatten()[None,:]
            pixels[4,:,i] = values
        self.p_s = pixels
        return self.p_s
    
    def create_I_from_ps(self):
        """Creating matrices from p_s especially during sampling"""
        dim_0, dim_1 = int(t.max(self.p_s[0,:,:]).item() + 1), int(t.max(self.p_s[1,:,:]).item() + 1)
        I = t.zeros(dim_0, dim_1, self.p_s.shape[2])
        for sl in range(0,I.shape[2]):
            I[:,:,sl] = self.p_s[4,:,sl].view(dim_0,dim_1)
        self.I = I
        return I 
        
       
    def create_p_r(self):
        """Function to calculate p_r the continuous voxel position in world coordinates R^(5xl*l,k) (x,y,z,0,value) - according to current transformation F"""
        #create mesh 
        x_lin,y_lin,k_lin = t.linspace(0,self.I.shape[0]-1,self.I.shape[0]), t.linspace(0,self.I.shape[1]-1,self.I.shape[1]), t.linspace(0,self.I.shape[2]-1,self.I.shape[2])
        #data points in voxel coordinates
        pixels = self.p_s[:4,:,0]
        
        #transform into reference coordinate system
        output = t.einsum('ijk,jl->ilk',self.F_inv,pixels)
        #switch order for consistency with 2d coordinates and create mes
        k_grid, x_grid, y_grid = t.meshgrid(k_lin, x_lin, y_lin)
        coordinates_3d = t.stack((t.flatten(x_grid), t.flatten(y_grid), t.flatten(k_grid)), dim = 0).int().numpy()
        
        #extract the values
        values = self.I[coordinates_3d][None,:]
        
        #bring output in 2d formate (x,y,z,0)...
        output = output.reshape(output.shape[0],output.shape[1]*output.shape[2])
        output = t.cat((output,values),0)
        self.p_r = output
        return self.p_r
    
    def corners(self):
        """Function to get the "corners" of the stack in world coordinates to estimate the target volume size"""
        p_r_max = t.max(self.p_r[:3,:], dim = 1).values
        p_r_min = t.min(self.p_r[:3,:], dim = 1).values
        corners = t.stack((p_r_min,p_r_max)).transpose(0,1)
        return corners

    def sample_from_volume(self, target_volume, dist_thresholde:float = 1.5):
        #distance threshold
        dist_threshold = 1.5
        #p_s_tilde scaled to image to sample to
        p_s_tilde_X = target_volume.scale_ps_tilde(self)
        #iterate over slices
        for sl in range(0,self.k):
            #get p_s_tilde and p_s of current slice and transpose
            p_s_tilde_t_complete = p_s_tilde_X[:,:,sl].transpose(0,1).float()
            p_s = self.p_s[:,:,sl]
            p_s_t = p_s.transpose(0,1).float()
            #calculate distance tensor
            distance_tensor = t.abs(p_s_t[:,:3].unsqueeze(0) - p_s_tilde_t_complete[:,:3].unsqueeze(1))
            #get indices of close pairs
            indices = t.where(t.all(distance_tensor < dist_threshold,2))
            length = list(indices[0].shape)[0]
            #get voxel intensities
            relevant_p_r = p_s_tilde_X[4, indices[0], sl]
            #get the relevant distances
            relevant_dist = distance_tensor[indices[0],indices[1],:]
            #calc the impact on pixel with PSF 
            gauss = utils.PSF_Gauss_vec(relevant_dist)
            value = t.multiply(gauss, relevant_p_r)
            #add values to corresponding pixesl
            for voxel in range(0,length):
                self.p_s[4,indices[1][voxel],sl] += value[voxel]
            print(f'slice {sl} sampled \n ')
        self.create_I_from_ps()
        print('I updated')
        
