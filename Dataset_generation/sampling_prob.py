import normflows as nf
import sys
import numpy as np
from matplotlib import pyplot as plt
import torch
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from datetime import datetime
num_cores = multiprocessing.cpu_count()

enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
grid_size=200

myJobID = int( sys.argv[1] )

## Definition of a simple model
K = 5
torch.manual_seed(0)

latent_size = 3 #number of dimensions
hidden_units = 128
hidden_layers = 5

flows = []
for i in range(K):
    flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units)]
    flows += [nf.flows.LULinearPermute(latent_size)]

# Set base model q0
q0 = nf.distributions.DiagGaussian(3, trainable=False)
    
# Construct flow model
model = nf.NormalizingFlow(q0=q0, flows=flows)
model = model.to(device)

####### Load models #########
model.load_state_dict(torch.load('Models/ref_model_3D_1000_7M_v5_light.pt',map_location=torch.device('cpu')))


####### Sampling  #########
xx, yy = torch.meshgrid(torch.linspace(0.02, 0.17, grid_size), torch.linspace(0, 1, grid_size))
kk=torch.ones((grid_size,grid_size))*(myJobID+0.02/0.00075)*0.00075
zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2), kk.unsqueeze(2)], 2).view(-1, 3).type(torch.float32)
    
processed_list = model.log_prob(zz).view(*xx.shape)
prob=torch.exp(processed_list.type(torch.float32)).view(*xx.shape).detach()
torch.save(prob, 'sampled_prob_0p2-1p7/new_sampled_prob_{}.pt'.format(myJobID))

