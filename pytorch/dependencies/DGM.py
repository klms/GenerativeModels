import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from VAE import Encoder, Decoder, VAE


class Classifier(nn.Module):
    def __init__(self, dims):
        super(Classifier, self).__init__()
        [x_dim, h_dim, y_dim] = dims
        self.h = nn.Linear(x_dim, h_dim)
        self.logits = nn.Linear(h_dim, y_dim)
    
    def forward(self, x):
        x = F.relu(self.h(x))
        x = F.softmax(self.logits(x))
        return x
    
class DGM(VAE):
    def __init__(self, dims, ratio):
        self.alpha = 0.1*ratio
        [x_dim, h_dim, z_dim, self.y_dim] = dims
        
        super(DGM, self).__init__([x_dim, h_dim, z_dim])
        self.encoder = Encoder([x_dim+self.y_dim, h_dim, z_dim])
        self.decoder = Decoder([z_dim+self.y_dim, list(reversed(h_dim)), x_dim])
        self.classifier = Classifier([x_dim, h_dim[-1], self.y_dim])
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x, y=None):
        logits = self.classifier(x)
        if y is None:
            return logits
        z, z_mu, z_logvar = self.encoder(torch.cat([x,y], dim=1))
        reconstruction = self.decoder(torch.cat([z,y], dim=1))
        return reconstruction, logits, (z, z_mu, z_logvar)
    
    def sample(self, z, y):
        y = y.type(torch.FloatTensor)
        x = self.decoder(torch.cat([z,y], dim=1))
        return x