import nflows as nf
import numpy as np
from matplotlib import pyplot as plt
import torch
import pandas as pd
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import os
import sys

def Flows_model(q0,K,hidden_units,hidden_layers,show_plot,latent_size):
    
    # Define flows
    torch.manual_seed(0)

    hidden_units = 128
    hidden_layers = 5

    flows = []
    for i in range(K):
        flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units)]
        flows += [nf.flows.LULinearPermute(latent_size)]
    
    # Construct flow model
    model = nf.NormalizingFlow(q0=q0, flows=flows)
    