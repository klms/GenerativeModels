import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class VAE_trainer(nn.Module):
    def __init__(self, model, objective, optimizer, cuda):
        super(VAE_trainer, self).__init__()
        self.model = model
        self.objective = objective
        self.optimizer = optimizer
        self.cuda = cuda
        if cuda:
            self.model = self.model.cuda()
    
    def _loss(self, x, y=None):
        x = Variable(x)
        if self.cuda:
            x=x.cuda()
        recon_x, (z, z_mu, z_logvar) = self.model(x)
        loss = self.objective(recon_x, x, z_mu, z_logvar)
        return torch.mean(loss)
    
    def train(self, labeled, unlabeled, n_epochs):
        for epoch in range(n_epochs):
            for unlabeled_image, _ in unlabeled:
                loss = self._loss(unlabeled_image)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            if (epoch+1)%10 == 0:
                print("Epoch: {}, loss:{:.3f}".format(epoch+1, loss.data[0]))

class Classifier_trainer(nn.Module):
    def __init__(self, model, classifier, cuda):
        super(Classifier_trainer, self).__init__()
        self.model = model
        self.classifier = classifier
        self.cuda = cuda
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr = 1e-3)
        if self.cuda:
            self.model = self.model.cuda()
            self.classifier = self.classifier.cuda()
    
    def _calculate_z(self, x):
        _, (z, _, _) = self.model(x)
        return z
    
    def _calculate_logits(self, z):
        logits = self.classifier(z)
        return logits
    
    def train(self, train_loader, validation_loader, n_epochs):
        for epoch in range(n_epochs):
            for trn_x, trn_y in train_loader:
                trn_x, trn_y = Variable(trn_x), Variable(trn_y)
                if self.cuda:
                    trn_x, trn_y = trn_x.cuda(), trn_y.cuda()
                logits = self._calculate_logits(self._calculate_z(trn_x))
                loss = F.cross_entropy(logits, trn_y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            if (epoch+1)%10==0:
                accuracy = []
                for val_x, val_y in validation_loader:
                    val_x = Variable(val_x)
                    if self.cuda:
                        val_x = val_x.cuda()
                        val_y = val_y.cuda()
                    logits=self._calculate_logits(self._calculate_z(val_x))
                    _, val_y_pred = torch.max(logits, 1)
                    accuracy += [torch.mean((val_y_pred.data == val_y).float())]
                    
                print("Epoch: {0:} loss: {1:.3f}, accuracy: {2:.3f}".format(epoch+1, loss.data[0], np.mean(accuracy)))