import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

eps = 1e-7

def KL_divergence_normal(mu, logvar):
    # KL[N(mu, var) || N(0,I)]
    return 0.5*(1. + logvar - mu**2 - torch.exp(logvar))

def binary_cross_entropy(x_hat, x):
    return -torch.sum((x * torch.log(x_hat + eps) + (1 - x) * torch.log((1 - x_hat) + eps)), dim=-1)
    #return F.binary_cross_entropy(x_hat, x, size_average=True)

def cross_entropy(logits, y):
    return -y*torch.log(logits + eps)
    
def discrete_uniform_prior(x):
    # CrossEntropy between categorical vector and uniform prior
    # x: categorical vector with Variable type
    # return: cross entropy with Variable type
    
    [batch_size, n_category] = x.size()
    uniform_prior = (1. / n_category) * torch.ones(batch_size, n_category)
    uniform_prior = F.softmax(uniform_prior)
    
    cross_entropy = -torch.sum(x * torch.log(uniform_prior + eps), dim=1)
    return -cross_entropy
        
class VariationalInference(nn.Module):
    def __init__(self, recon_prob, KL_div):
        super(VariationalInference, self).__init__()
        self.recon_prob = recon_prob
        self.KL_div = KL_div
    
    def forward(self, x_hat, x, mu, logvar):
        log_likelihood = self.recon_prob(x_hat, x)
        KL_div = self.KL_div(mu, logvar)
        #KL_div = torch.sum(self.KL_div(mu, logvar))
        return torch.mean(log_likelihood) - KL_div

class VariationalInference_with_labels(VariationalInference):
    def __init__(self, recon_prob, KL_div):
        super(VariationalInference_with_labels, self).__init__(recon_prob, KL_div)

    def forward(self, x_hat, x, y, latent):
        log_prior_y = self.prior_y(y)
        log_likelihood = self.recon_prob(x_hat, x)
        KL_div = [torch.sum(self.KL_div(mu, logvar), dim=-1) for _, mu, logvar in latent]
        return log_likelihood + log_prior_y + sum(KL_div)