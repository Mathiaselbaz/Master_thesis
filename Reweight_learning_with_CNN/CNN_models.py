import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
from torchvision.models import ResNet50_Weights

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Define a custom PyTorch module for a trained ResNet model with extra Poisson dimension
class Trained_ResNet_Poisson(nn.Module):
    def __init__(self, n_beta):
        super(Trained_ResNet_Poisson, self).__init()
        
        # Load the ResNet-50 model with pre-trained weights
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Replace the fully connected layer with an identity layer
        resnet.fc = Identity()
        
        # Create a sequential model with the modified ResNet
        self.resnet = nn.Sequential(resnet)
        
        # Define activation functions and layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.LazyLinear(2048)
        self.fc2 = nn.Linear(2048, n_beta)
        
        # Initialize the bias of fc2 with uniform values between 0 and 2
        nn.init.uniform_(self.fc2.bias, 0, 2)

    def forward(self, x):
        outputs = []
        # Loop through each Poisson fluctuation in the input tensor
        for i in range(x.shape[1]):
            y = x[:, i, :, :, :]
            y.requires_grad_(True)
            # Pass the data through the ResNet model
            y = self.resnet(y)
            # Flatten the output
            y = torch.flatten(y, 1)
            # Apply activation and fully connected layers
            y = self.relu(self.fc1(y))
            y = self.fc2(y)
            # Append the result to the outputs list
            outputs.append(y)
        # Stack the outputs along the specified dimension
        x = torch.stack(outputs, 1)
        
        return x

