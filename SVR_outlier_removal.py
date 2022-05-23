
from turtle import forward
from typing import Tuple
#from sklearn.covariance import log_likelihood
import torch as t
import numpy as np

class Outlier_Removal_Voxels(t.nn.Module):
    """
    Class for outlier removal of voxels, using Gaussian zero mean pdf for inliers 
    and uniform distribution with densitiy m for outliers.
    Parameter fitting using EM-algorithm.
    """
    def __init__(self, e:t.tensor) -> None:
        """
        Construcor of outlier using EM algorithm

        Args:
            e (t.tensor): Error image
        """
        super().__init__()
        #calculated constant density of outlier distribution as reciprocal of the range of e
        range_e = (t.max(e) - t.min(e)).item()
        self.m = 1 / range_e
        #use zero mean of gaussian
        self.mu = 0
    
    def gaussian(self, e:t.tensor, variance:t.tensor)->t.tensor:
        """
        calculate zero mean Gaussian with current sigma_squared

        Args:
            e (t.tensor): error image
            sigma_squared (t.tensor): current variance

        Returns:
            t.tensor: output of Gaussian 
        """
        factor = t.div(1,t.sqrt(variance * 2 * t.pi))
        argument = t.div((e - self.mu), t.sqrt(variance))
        exponential = t.exp(-0.5 * t.pow((argument),2) )
        return factor * exponential

    def expectation(self, e:t.tensor, variance:float, c:float)->t.tensor:
        """updated likelihood of being an inlier with current parameters c and sigma_squared

        Args:
            e (t.tensor): error image
            sigma_squared (float): _description_
            c (float): mixing coefficient of in and outlier

        Returns:
            t.tensor: probability image of being inlier
        """
        gauss =  self.gaussian(e,variance)
        denominator = gauss * c + self.m * (1 - c)
        return t.div(gauss * c, denominator)

    def maximization(self, e:t.tensor, p:float)->tuple:
        """updates parameters during EM-algorithm

        Args:
            e (t.tensor): error_image
            p (float): probabability iamge of being inlier

        Returns:
            float: updated parameters c and sigma_squared
        """
        nominator = t.sum(t.mul(p,t.pow(e,2)),[0,1])
        variance = t.div(nominator,t.sum(p,[0,1]))
        c = t.div(t.sum(p,[0,1]), t.numel(p) )
        return variance, c

    def forward(self,e:t.tensor)->t.tensor:
        """
        EM-algorithm for outlier removal

        Args:
            e (t.tensor): error image

        Returns:
            t.tensor: inlier probabilities
        """
        avg_likelihoods = []
        likelihood_image_old = t.zeros_like(e)
        avg_delta = 1
        iterations = 0
        variance = t.tensor(1)
        c = t.tensor(0.5)
        while(avg_delta > 0.001  and iterations < 1000 ):
            p = self.expectation(e,variance, c)
            variance, c = self.maximization(e,p)
            #log_likelihood_image = t.log(p)

            avg_delta =  t.mean(t.abs(likelihood_image_old - p)).item()

            likelihood_image_old = p
            avg_likelihoods.append(t.mean(p))
            #print(f'average delta: {avg_delta}')
            iterations = iterations + 1
        return p

class Outlier_Removal_Slices_cste(t.nn.Module):
    def __init__(self, red_voxel_prob:t.tensor) -> None:
        """
        Construcor of outlier using EM algorithm

        Args:
            red_voxel_prob (t.tensor): reduced voxel probabilities to slices
        """
        super().__init__()
        #calculated constant density of outlier distribution as reciprocal of the range of e
        range_e = len(red_voxel_prob)
        self.m = 1 / range_e 

    def gaussian(self, red_voxel_prob:t.tensor, variance:t.tensor, mu:t.tensor)->t.tensor:
        """
        calculate zero mean Gaussian with current sigma_squared

        Args:
            red_voxel_prob (t.tensor): reduced voxel probabilities to slices
            sigma_squared (t.tensor): current variance for inliers
            mu (t.tensor): current mean of Gaussian for inliers

        Returns:
            t.tensor: output of Gaussian 
        """
        factor = t.div(1,t.sqrt(variance * 2 * t.pi))
        argument = t.div((red_voxel_prob - mu), t.sqrt(variance))
        exponential = t.exp(-0.5 * t.pow((argument),2) )
        return factor * exponential
    
    def expectation(self, red_voxel_prob:t.tensor, variance:float, mu:t.tensor, c:float)->t.tensor:
        """updated likelihood of being an inlier with current parameters c and sigma_squared

        Args:
            red_voxel_prob (t.tensor): reduced voxel probabilities to slices
            sigma_squared (float): _description_
            c (float): mixing coefficient of in and outlier

        Returns:
            t.tensor: probability image of being inlier
        """
        gauss =  self.gaussian(red_voxel_prob,variance, mu)
        denominator = gauss * c + self.m * (1 - c)
        return t.div(gauss * c, denominator)
    
    def maximization(self, red_voxel_prob:t.tensor, p:float)->tuple:
        """updates parameters during EM-algorithm

        Args:
            red_voxel_prob (t.tensor): reduced voxel probabilities to slices
            p (float): probabability iamge of being inlier

        Returns:
            float: updated parameters c and sigma_squared
        """
        responsibility_in = t.sum(p)
        mu = t.sum(t.mul(red_voxel_prob, p)) / responsibility_in
        nominator = t.sum(t.mul(p,t.pow(red_voxel_prob-mu,2)))
        variance = t.div(nominator,responsibility_in)
        c = t.div(responsibility_in, t.numel(p) )
        return variance, mu, c
    
    def forward(self,red_voxel_prob:t.tensor)->t.tensor:
        """
        EM-algorithm for outlier removal

        Args:
            red_voxel_prob (t.tensor): reduced voxel probabilities to slices

        Returns:
            t.tensor: inlier probabilities
        """
        avg_likelihoods = []
        likelihood_image_old = t.zeros_like(red_voxel_prob)
        avg_delta = 1
        iterations = 0
        variance = t.tensor(1.0)
        mu = t.tensor(1.0)
        c = t.tensor(0.5)
        while(avg_delta > 0.001  and iterations < 1000 ):
            p = self.expectation(red_voxel_prob,variance, mu, c)
            variance, mu, c = self.maximization(red_voxel_prob,p)
            #log_likelihood_image = t.log(p)

            avg_delta =  t.mean(t.abs(likelihood_image_old - p)).item()

            likelihood_image_old = p
            avg_likelihoods.append(t.mean(p))
            #print(f'average delta: {avg_delta}')
            iterations = iterations + 1
        return p   



class Outlier_Removal_Slices(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def gaussian(self, red_voxel_prob:t.tensor, variance:t.tensor, mu:t.tensor)->t.tensor:
        """
        calculate zero mean Gaussian with current sigma_squared

        Args:
            e (t.tensor): error image
            sigma_squared (t.tensor): current variance

        Returns:
            t.tensor: output of Gaussian 
        """
        factor = t.div(1,t.sqrt(variance * 2 * t.pi))
        argument = t.div((red_voxel_prob - mu), t.sqrt(variance))
        exponential = t.exp(-0.5 * t.pow((argument),2) )
        return factor * exponential

    def expectation(self, red_voxel_prob:t.tensor, var_in:float, mu_in:float, var_out:float, mu_out:float, c:float)->t.tensor:
        inlier_gauss = self.gaussian(red_voxel_prob,var_in, mu_in)
        outlier_gauss = self.gaussian(red_voxel_prob,var_out, mu_out)
        return c * inlier_gauss / (c*inlier_gauss + (1-c) * outlier_gauss)
    
    def maximization(self, red_voxel_prob:t.tensor, p_slice:t.tensor) -> tuple:
        #for two class problem respoonsibilities simplify
        responsibility_in = t.sum(p_slice)
        responsibility_out = t.sum(1 - p_slice)

        mu_in = t.sum(t.mul(red_voxel_prob, p_slice)) / responsibility_in
        mu_out = t.sum(t.mul(red_voxel_prob,(1 - p_slice))) / responsibility_out

        var_in = t.sum(p_slice * t.pow((red_voxel_prob - mu_in),2)) / responsibility_in
        var_out = t.sum((1 - p_slice) * t.pow((red_voxel_prob - mu_out),2)) / responsibility_out

        c = responsibility_in / (responsibility_in + responsibility_out)

        return var_in, mu_in, var_out, mu_out, c

    def forward(self, red_voxel_prob:t.tensor) -> t.tensor:
        """_summary_

        Args:
            red_voxel_prob (t.tensor): reduced voxel probability

        Returns:
            t.tensor: inlier probability of slices
        """
        slice_likelihoods_old = t.zeros_like(red_voxel_prob)
        avg_delta = 1
        iterations = 0
        var_in, var_out = t.tensor(1),t.tensor(1)
        mu_in, mu_out = t.tensor(0),t.tensor(0)
        c = 0.5

        p_slices = red_voxel_prob
        var_in, mu_in, var_out, mu_out, c = self.maximization(red_voxel_prob,p_slices)

        while(avg_delta > 0.003  and iterations < 200 ):
            p_slices = self.expectation(red_voxel_prob, var_in, mu_in, var_out, mu_out, c)
            var_in, mu_in, var_out, mu_out, c = self.maximization(red_voxel_prob,p_slices)
            avg_delta =  t.mean(t.abs(slice_likelihoods_old - p_slices)).item()
            slice_likelihoods_old = p_slices
            iterations = iterations + 1
        return p_slices








