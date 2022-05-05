
from turtle import forward
from typing import Tuple
from sklearn.covariance import log_likelihood
import torch as t
import numpy as np

class outlier_removal(t.nn.Module):
    def __init__(self, e:t.tensor) -> None:
        """
        Constructr of outlier using EM algorithm

        Args:
            e (t.tensor): Error image
        """
        super().__init__()
        #calculated constant density of outlier distribution as reciprocal of the range of e
        range_e = (t.max(e) - t.min(e)).item()
        self.m = 1 / range_e
        #use zero mean of gaussian
        self.mu = 0
    
    def gaussian(self, e:t.tensor, sigma_squared:t.tensor)->t.tensor:
        """
        calculate zero mean Gaussian with current sigma_squared

        Args:
            e (t.tensor): error image
            sigma_squared (t.tensor): current variance

        Returns:
            t.tensor: output of Gaussian 
        """
        factor = t.div(1,t.sqrt(sigma_squared * 2 * t.pi))
        argument = t.div((e - self.mu), t.sqrt(sigma_squared))
        exponential = t.exp(-0.5 * t.pow((argument),2) )
        return factor * exponential

    def expectation(self, e:t.tensor, sigma_squared:float, c:float)->t.tensor:
        """updated likelihood of being an inlier with current parameters c and sigma_squared

        Args:
            e (t.tensor): error image
            sigma_squared (float): _description_
            c (float): mixing coefficient of in and outlier

        Returns:
            t.tensor: probability image of being inlier
        """
        gauss =  self.gaussian(e,sigma_squared)
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
        sigma_squared = t.div(nominator,t.sum(p,[0,1]))
        c = t.div(t.sum(p,[0,1]), t.numel(p) )
        return sigma_squared, c

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
        sigma_squared = t.tensor(1)
        c = t.tensor(0.5)
        while(avg_delta > 0.001  and iterations < 100 ):
            p = self.expectation(e,sigma_squared, c)
            sigma_squared, c = self.maximization(e,p)
            #log_likelihood_image = t.log(p)

            avg_delta =  t.mean(t.abs(likelihood_image_old - p)).item()

            likelihood_image_old = p
            avg_likelihoods.append(t.mean(p))
            #print(f'average delta: {avg_delta}')
            iterations = iterations + 1
        return p
        





