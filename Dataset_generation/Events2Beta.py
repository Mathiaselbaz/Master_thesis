# Import necessary libraries
import numpy as np
from matplotlib import pyplot as plt
import torch
import pandas as pd
import os
import sys
from scipy.optimize import minimize

# Get the job ID from command line arguments
myJobID = int(sys.argv[1])

# Enable CUDA if available
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

# Define constants
num_samples_per_beta = 100000
n_beta = 3
n_poisson = 1000

# Set the directory for the dataset
data_dir = 'Events/2023-05-18_MC_sample_3bins_200MeV-1p7GeV_Unif0-2_1128M_equifilled_rowbeta_100k'

# Load the dataset
dataset = torch.load(data_dir + '.pt')[num_samples_per_beta * (myJobID - 1) * 100:num_samples_per_beta * myJobID * 100, :].clone()

# Calculate the number of histograms in the dataset
n_hist = len(dataset) // num_samples_per_beta

# Create an array to store beta values
Betas = torch.zeros(n_hist * n_poisson, n_beta)

# Define a function to calculate the chi-square
def chi_square(beta, hist, n_beta, nom_prob):
    prob_grid = torch.sum(torch.Tensor(beta).unsqueeze(0).unsqueeze(0).expand(200, 200, n_beta).to(device) * nom_prob.to(device), dim=-1) * 50000
    # Calculate the residuals
    chi2 = torch.sum((hist.to(device) - prob_grid) ** 2)
    return chi2

# Define a function to optimize for beta values
def chi_2(hist, n_beta, nom_prob):
    initial_beta = [1, 1, 1]
    Beta = torch.zeros(hist.shape[0], n_beta)
    X = hist.clone()

    def objective(beta):
        return chi_square(beta, X, n_beta, nom_prob)

    # Perform the optimization
    result = minimize(objective, initial_beta, method='Nelder-Mead')
    Beta = torch.Tensor(result.x)
    return Beta

# Define constants related to energy bins
nb_energy = 199
bin_length = int(nb_energy / n_beta) + 1
nom_prob = torch.zeros(200, 200, n_beta)
bin_edge = [0, 55, 79, 199]  # squared interval 0p2-1p7

# Load and accumulate probability data into nom_prob
for i in range(n_beta):
    for j in range(bin_edge[i], bin_edge[i + 1]):
        p = torch.load('sampled_prob_0p2-1p7/new_sampled_prob_' + str(j + 1) + '.pt')
        nom_prob[:, :, i] += p

# Normalize nom_prob
nom_prob = (nom_prob / torch.sum(nom_prob))

# Iterate over each histogram in the dataset
for i in range(n_hist):
    data_idx = i * num_samples_per_beta
    x = dataset[data_idx:data_idx + num_samples_per_beta, :].clone()
    
    # Extract non-zero indices from the data
    nonzero_indices = torch.nonzero(torch.sum(x, dim=1))
    x_nonzero = x[nonzero_indices.squeeze()].clone()
    
    # Create a histogram from the non-zero data
    hist = torch.histogramdd(x_nonzero, 200, range=[0.02, 0.17, 0., 1.])[0]
    
    # Calculate beta values and store in Betas
    for j in range(n_poisson):
        Betas[i * n_poisson + j, :] = chi_2(torch.poisson(hist), n_beta, nom_prob)

# Save the beta values to a file
torch.save(Betas, 'logbeta/beta_' + str(myJobID) + '.pt')


    







