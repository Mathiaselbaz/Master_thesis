import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
from torchvision.models import ResNet50_Weights

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    
class Trained_ResNet_4_Linear(nn.Module):
    def __init__(self,n_beta):
        super(Trained_ResNet_4_VAE, self).__init__()

        resnet=models.resnet50(weights=ResNet50_Weights.DEFAULT) #load upto the classification layers except first conv layer
        resnet.fc=Identity()
        self.resnet =nn.Sequential(resnet)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.8)
        self.fc1 = nn.LazyLinear(128)
        self.fc3 = nn.Linear(128, 3*n_beta)
        self.fc3.bias.data=torch.Tensor([1,1,1,-2,-2,-2,-10,-10,-10])
        
    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x,1)
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.fc3(x)
        return x
    
class Trained_ResNet_4_NSF(nn.Module):
    def __init__(self,n_context):
        super(Trained_ResNet_4_NSF, self).__init__()

        resnet=models.resnet50(weights=ResNet50_Weights.DEFAULT) #load upto the classification layers except first conv layer
        resnet.fc=Identity()
        self.resnet =nn.Sequential(resnet)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.8)
        self.fc1 = nn.LazyLinear(n_context)
        
    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x,1)
        x = self.relu(self.fc1(self.dropout(x)))
        return x
        
        
