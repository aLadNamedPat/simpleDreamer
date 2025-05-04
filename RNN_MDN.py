import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class RNN_MDN(nn.Module):
    def __init__(
        self, 
        input_size : int,
        hidden_size : int,
        num_gaussians : int,
        num_layers : int = 1,
    ):
        """
        Implementation of MDN-RNN using LSTMs
        """
        super(RNN_MDN, self).__init__()
        self.input_size = input_size
        self.num_gaussians = num_gaussians
        self.hidden_size = hidden_size

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)

        self.firstLayer = nn.Linear(input_size, 40)
        self.mu = nn.Linear(input_size,  input_size * num_gaussians)
        self.var = nn.Linear(input_size, input_size * num_gaussians)
        self.pi = nn.Linear(input_size,  input_size * num_gaussians)
        self.leakyReLU = nn.LeakyReLU()

    def MDN(
        self,
        x
    ):
        """
        MASSIVE HELP FROM: https://www.katnoria.com/mdn/
        ALSO HELP FROM: https://github.com/sksq96/pytorch-mdn/blob/master/mdn-rnn.ipynb

        MDN is a 'mixture of gaussians' where we learn distributions rather than learn direct mappings

        Similar to the reparametrization trick from VAEs, we seek to learn the probability distribution for each target and sample the predicted output

        We have L input features, K gaussians to model the multimodal distribution

        We seek to learn the distribution p(y | x) = \sum_k{ \pi_k(x) \N(y | \mu_k(x), \var_k(x))}

        So let utilize non-linear models to predict the following: \pi_k, \mu_k and \var_k

        Dimensionality of each output:
        1. pi_k: K * input_features
        2. mu_k : K * input_features
        3. var_k : K * input_features
        
        Chose not to use isotropic gaussians in this implementation. Felt kind of limiting.
        """

        x = self.firstLayer(x)
        x = self.leakyReLU(x)
        # Output for all of these are 1 dimensional L * K tensors. We want to convert this into a vector of dimensionality [x.shape[1], L, K]
        mu = self.mu(x)
        mu = mu.view(-1, x.shape[-1], self.num_gaussians, self.input_size)

        var = torch.exp(self.var(x)) # Need to have a positive variance here
        var = var.view(-1, x.shape[-1], self.num_gaussians, self.input_size)

        pi = self.pi(x)
        pi = pi.view(-1, x.shape[-1], self.num_gaussians, self.input_size)
        pi = F.softmax(pi, 2) 

        return mu, var, pi
    
    def find_pdf_normal(
        self,
        y, 
        mu,
        var
    ):
        """
        Method to calculate the probability of obtaining a given y value for a normal distribution given by

        f(x) = \frac{e^{\frac{-(x - \mu)^2}{2 \var}}}{\sqrt{2\pi\var}}
        """
        numerator = torch.exp(-(torch.subtract(y, mu)**2) / (2 * var))
        denominator = torch.sqrt(2 * np.pi * var)

        return numerator / denominator
    
    def MDN_loss(
        self, 
        x,
        y_real,
    ):
        """
        Our loss method is defined by the following equation:
        L(w) = \frac{-1}{N} \sum_{n = 1}^_{N} { \log{\sum_{k}{\pi_k(x_n, w)\N(y_n|\mu_k(x_n, w), \var(x_n, w))}}}
        """

        mu, var, pi = self.MDN(x)
        pdf = self.find_pdf_normal(y_real, mu, var)
        val = -torch.log(torch.sum(pdf * pi))
        return val / -x.shape[-1]
    
    def get_initial_hidden(
        self
    ):
        return torch.zeros(self.hidden_size)
    
    def forward(
        self,
        x,
        h
    ):
        """
        Method for prediction. Only need the hidden layer and the mu and var prediction for the output prediction.

        """
        y, hidden = self.lstm(x, h)
        mu, var, pi = self.MDN(y)

        return (mu, var), hidden