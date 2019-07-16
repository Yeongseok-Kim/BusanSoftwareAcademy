import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.layer=nn.Sequential(
            nn.Linear(784,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        out=x.view(x.size(0),-1)
        out=self.layer(out)
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.layer=nn.Sequential(
            nn.Linear(64,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,784),
            nn.Tanh()
        )
    def forward(self,x):
        out=self.layer(x)
        return out