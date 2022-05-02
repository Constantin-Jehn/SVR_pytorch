
from turtle import forward
from typing import Tuple
from sklearn.covariance import log_likelihood
import torch as t
import numpy as np

class outlier_removal(t.nn.Module):
    def __init__(self, e:t.tensor) -> None:
        super().__init__()
        #calculated constant density of outlier distribution as reciprocal of the range of e
        range_e = t.max(e) - t.min(e)
        self.m = 1 / range_e
        #use zero mean of gaussian
        self.mu = 0
    
    def gaussian(self, e:t.tensor, sigma_squared) -> t.t.tensor:
        factor = 1/(np.sqrt(self.sigma_squared * 2 * np.pi))
        argument = (e - self.mu) / np.sqrt(sigma_squared)
        exponential = t.exp(-0.5 * t.pow((argument),2) )
        return factor * exponential

    def expectation(self, e:t.tensor, sigma_squared:float, c:float) -> t.tensor:
        """
        updated likelihood of being an inlier with current parameters c and sigma_squared
        """
        gauss =  self.gaussian(e,sigma_squared)
        denominator = gauss * c + self.m * (1 - c)
        return t.div(gauss * c, denominator)

    def maximization(self, e:t.tensor, p:t.tensor) -> Tuple(float,float):
        """
        updates parameters during EM-algorithm
        sigma_squared: standard deviation of gaussian
        c: mixing coefficients between in and outliers
        """
        sigma_squared = t.sum(t.mul(p,t.pow(e,2)),[0,1])
        c = t.div(t.sum(p,[0,1]), t.numel(p) )
        return sigma_squared, c

    def forward(self,e:t.tensor) -> t.tensor:
        avg_log_likelihoods = []
        log_likelihood_image_old = t.zeros_like(e)
        avg_log_delta = 1
        iterations = 0
        sigma_squared = 1
        c = 0.5
        while(avg_log_delta > 0.001  and iterations < 100 ):
            p = self.expectation(e,sigma_squared, c)
            sigma_squared, c = self.maximization(e,p)
            log_likelihood_image = t.log(p)

            avg_log_delta =  t.mean(log_likelihood_image_old - log_likelihood_image)

            log_likelihood_image_old = log_likelihood_image
            avg_log_likelihoods.append(t.mean(log_likelihood_image))

        return p
        





