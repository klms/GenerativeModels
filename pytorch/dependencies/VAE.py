import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from Sampling import Gaussian_sample

class Encoder(nn.Module):
    def __init__(self, dims, dataset='mnist'):
        super(Encoder, self).__init__()
        [x_dim, h_dim, z_dim] = dims
        neurons = [x_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]
        self.h= nn.ModuleList(modules=linear_layers)
        self.sample = Gaussian_sample(h_dim[-1], z_dim)
    
    def forward(self, x):
        for i, next_layer in enumerate(self.h):
            x = next_layer(x)
            if i < len(self.h) - 1:
                x = F.softplus(x)
        return self.sample(x)
    
    
class Decoder(nn.Module):
    def __init__(self, dims, dataset='mnist'):
        super(Decoder, self).__init__()
        [z_dim, h_dim, x_dim] = dims
        neurons = [z_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]
        self.h = nn.ModuleList(modules=linear_layers)
        self.output = nn.Linear(h_dim[-1], x_dim)
        
    def forward(self, x):
        for i, next_layer in enumerate(self.h):
            x=F.softplus(next_layer(x))
        return F.sigmoid(self.output(x))
    
    
class VAE(nn.Module):
    def __init__(self, dims):
        super(VAE, self).__init__()
        [x_dim, h_dim, self.z_dim] = dims
        self.encoder = Encoder([x_dim, h_dim, self.z_dim])
        self.decoder = Decoder([self.z_dim, list(reversed(h_dim)), x_dim])
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_normal(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, (z, mu, logvar)
    
    def sample(self, z):
        return self.decoder(z)
    
class Classifier(nn.Module):
    def __init__(self, dims, dataset='mnist'):
        super(Classifier, self).__init__()
        [z_dim, h_dim, n_class] = dims
        neurons = [z_dim, *h_dim, n_class]
        linear_layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]
        self.h = nn.ModuleList(modules=linear_layers)
        self.output = nn.Linear(dims[-1], n_class)
    
    def forward(self, z):
        for i, next_layer in enumerate(self.h):
            z = next_layer(z)
            if i < len(self.h) - 1:
                z = F.relu(z)
        return F.softmax(self.output(z))