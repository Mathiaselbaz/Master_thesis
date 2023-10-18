import torch
import numpy as np
from matplotlib import pyplot as plt
import time


def Prob_grid(prob_dir, nb_energy, beta_range, nb_beta):
    # Create a tensor 'Betas' with random values between 0 and 'beta_range'.
    Betas = torch.rand(nb_beta) * beta_range
    Betas[Betas < 0] = 0
    # Calculate the bin length
    bin_length = int(np.round(nb_energy / nb_beta))
    # Create tensors the probability tensor filled with zeros.
    A = torch.zeros(200, 200, nb_beta)
    # Define bin edges for processing.
    bin_edge = [0, 55, 79, 199]
    for i in range(nb_beta):
        for j in range(bin_edge[i], bin_edge[i+1]):
            # Load probability data from files and scale it by 'Betas'.
            B = torch.load(prob_dir+'/new_sampled_prob_'+str(j+1)+'.pt') * Betas[i]
            # Update 'A' by adding the scaled probability values.
            A[:, :, i] += B
    A = A / (torch.max(A).item())
    return Betas, A

# Define constants and initialize variables.
nb_energy = 199
nb_beta = 3
beta_range = 2
nb_sample = 600000
nb_sim_sample = 10000
grid_size = 200
m=0
n=0
k=0
MC_sample=torch.zeros((nb_sample,3)) 

# Create nb_sample/nb_sim_sample datasets corresponding to different betas with nb_real_sample samples (and nb_sim_sample-nb_real_sample fake samples filled with 0)
while m < nb_sample :
    Betas,prob_grid = Prob_grid('sampled_prob_0p2-1p7', nb_energy, beta_range, nb_beta)
    # Calculate the number of true samples 
    nb_real_sample=np.round(torch.dot(Betas/2,torch.tensor([1/3,1/3,1/3])).item()*nb_sim_sample)
    n=0
    while n < nb_sim_sample :        
        if n < nb_real_sample :
            # Accept-Reject Monte Carlo Generator
            obs_sample = np.random.rand(3)* np.array([0.15, 1, 0.15])
            idx = np.trunc(obs_sample*np.array([1/0.15*(grid_size), grid_size, 1/0.15*(nb_beta)])).astype('int32')
            p=np.random.rand(1)
            if p.item() < prob_grid[idx[0],idx[1],idx[2]].item() :
                MC_sample[m,:]=torch.from_numpy(obs_sample+np.array([0.02, 0, 0.02]))
                n=n+1
                m=m+1
        else: 
            MC_sample[m,:]=torch.Tensor([0, 0, 0])
            n=n+1
            m=m+1

torch.save(MC_sample[:,:2].clone(),'Events/'+str(datetime.now())+'MC_sample_'+str(int(nb_sample/1000000))+'M_'+str(nb_beta)+'bins.pt')       
        
