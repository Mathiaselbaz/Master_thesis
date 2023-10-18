################ Imports packages and set up ddp ############################

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision.models import ResNet50_Weights
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
from CNN_models import Trained_ResNet_Poisson

import os
import sys
import torchvision.transforms as tr

enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    

## Creation of Training and validation samples

class MC(Dataset):
    def __init__(self, data_dir, start, num_samples, num_beta = 10, num_samples_per_beta = 50000 , nb_bins= 200, n_poisson = 5):
        self.data = torch.load(data_dir)[start*num_samples_per_beta:(start+num_samples)*num_samples_per_beta,:].clone()
        self.x_mean = self.data.mean()
        self.x_std = self.data.std()
        self.num_beta =num_beta
        self.num_samples_per_beta = num_samples_per_beta
        self.nb_bins =nb_bins
        self.n_poisson = n_poisson

    def __len__(self):
        return self.data.size(0)//self.num_samples_per_beta

    def __getitem__(self, idx):
        data_idx = idx*self.num_samples_per_beta
        x = self.data[data_idx:data_idx+self.num_samples_per_beta,:].clone()
        nonzero_indices = torch.nonzero(torch.sum(x, dim=1))
        x_nonzero = x[nonzero_indices.squeeze()].clone()
        hist = torch.histogramdd(x_nonzero, bins=self.nb_bins, range=[0.02, 0.17, 0., 1.])[0]
        del(x_nonzero,x,nonzero_indices)
        fluctuated_hist = torch.zeros((self.n_poisson, 3, self.nb_bins, self.nb_bins))
        Av=torch.nn.AvgPool2d(kernel_size=5,stride=1,padding=2)
        for i in range(self.n_poisson):
            fluctuated_hist[i, :, :, :] = torch.poisson(hist).unsqueeze(0).repeat(3, 1, 1)
            fluctuated_hist[i, :, :, :] = fluctuated_hist[i, :, :, :]/10  
        return fluctuated_hist

# Early-stopper class
class EarlyStopper:
    def __init__(self, patience=20, min_delta=0.005):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = 1
        
    def save_model(self, validation_loss, model, index):
        if validation_loss == self.min_validation_loss:
            print(f'Model saved at epoch ' + str(index+1))
            torch.save(model.state_dict(), 'Models/CNN_Model.pt')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
# Loss class for the training   
class Loss_CNN_VAE(nn.Module):
    def __init__(self, n_bin,A,delta):
        super().__init__()
        self.n_bin = n_bin
        self.data0 = A
        self.delta=delta
        self.Plot=False

    def forward(self, x, beta,dev, train_or_val):
        # Add optionnal averaging in the histogram to smooth learning
        Av=torch.nn.AvgPool2d(kernel_size=7,stride=1,padding=3)
        output = torch.zeros_like(x)  # Create a new tensor to store the output
        for k in range(x.shape[0]):
            output[k,:,:,:,:] = Av(x[k,:,:,:,:])
            
        nb_energy = 199
        nb_beta = 3
        bin_length = int(nb_energy/nb_beta)+1
        self.data0=self.data0.to(dev)
        B=torch.zeros((x.shape[0],x.shape[1],200,200,nb_beta)).to(dev)
        # Compute the true histogram
        for k in range(x.shape[0]):
            for l in range(x.shape[1]):
                for i in range (nb_beta):
                    B[k,l,:,:,i] = self.data0[:,:,i].to(dev)*beta[k,l,i]      
        prob_grid=torch.sum(B,4).unsqueeze(2).repeat(1, 1, 3, 1, 1)*5000
        
        #Compute the difference between true and predicted histogram
        diff = torch.abs(x-prob_grid)
        mse = torch.mean(torch.sum(diff**2,dim=1,keepdim=False))
        return (mse)

# Trainer class
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        gpu_id: int,
        batch_size: int,
        loss_function
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler=scheduler
        self.model = DDP(model, device_ids=[gpu_id])
        self.Train_loss=[]
        self.Val_loss=[]
        self.Train_loss_batch=[]
        self.Val_loss_batch=[]
        self.R2=[]
        self.early_stopper = EarlyStopper()
        self.loss_t=0
        self.loss_v=0
        self.batch_size=batch_size
        self.loss_function = loss_function
        self.Betas = []

    def _run_batch(self, source):
        self.optimizer.zero_grad()
        predictions = self.model(source)
        loss = self.loss_function(source, predictions,self.gpu_id, 'train')
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.loss_t += loss.item()
        if loss.item()<1:
            self.Train_loss_batch.append(loss.item())
    
    def _run_batch_val(self, val):
        predictions = self.model(val)
        self.Betas.append(predictions.detach().to('cpu').numpy())
        loss = self.loss_function(val, predictions,self.gpu_id, 'val')
        self.loss_v += loss.item()
        self.Val_loss_batch.append(loss.item())           
        
    def _run_epoch(self, epoch):
        self.Betas=[]
        b_sz = self.batch_size
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_loader)}")
        self.train_loader.sampler.set_epoch(epoch)
        self.val_loader.sampler.set_epoch(epoch)
        self.loss_t = 0.0
        self.loss_v = 0.0
        for source in self.train_loader:
            source = source.to(self.gpu_id)
            self._run_batch(source)
        for val in self.val_loader:
            val = val.to(self.gpu_id)
            self._run_batch_val(val)
        self.Train_loss.append(self.loss_t/ len(self.train_loader))
        self.Val_loss.append(self.loss_v/ len(self.val_loader))      

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            # Early Stopping
            ES = self.early_stopper.early_stop(self.loss_v)
            if ES:             
                break
            self.early_stopper.save_model(self.loss_v, self.model, epoch)
            # Prediction R2 and losses
            plt.figure()
            ax1 = plt.subplot()
            l1, =ax1.plot(self.Train_loss, label='Training Loss')
            l2, =ax1.plot(self.Val_loss, label='Validation Loss')
            plt.legend([l1, l2], ['Training Loss','Validation Loss'], loc='center right')
            plt.savefig('Plots/Loss_CNN_VAE.png')
            plt.show()
            torch.save(self.Betas,'Plots/Betas_400MeV-1GeV7.pt')

            

def main(rank: int, world_size: int, total_epochs: int, num_training: int, num_validation: int, batch_size: int, n_poisson: int):
    ddp_setup(rank, world_size)

    # Define the datasets
    data_dir = 'Events/2023-05-18_MC_sample_3bins_200MeV-1p7GeV_Unif0-2_1128M_equifilled_rowbeta_100k.pt'
    train_dataset = MC(data_dir = data_dir, start = 1, num_samples = num_training, num_beta = 3, num_samples_per_beta = 100000, nb_bins= 200, n_poisson = n_poisson)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False,pin_memory=True,drop_last=False, sampler=DistributedSampler(train_dataset))
    val_dataset = MC(data_dir = data_dir, start = num_training+1, num_samples = num_validation, num_beta = 3, num_samples_per_beta = 100000, nb_bins= 200, n_poisson = n_poisson)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, pin_memory=True,drop_last=False,sampler=DistributedSampler(val_dataset))
    
    #Define the model
    model = Trained_ResNet_Poisson(3)
    input_data = torch.randn((batch_size,n_poisson,3,200,200))
    _ = model(input_data)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.97)
    
    #Compute the nominal probability
    nb_energy = 199
    nb_beta = 3
    A=torch.zeros(200,200,nb_beta)
    bin_length = int(nb_energy/nb_beta)+1
    bin_edge=[0, 55, 79, 199] #energy bin edges
    for i in range (nb_beta):
        for j in range(bin_edge[i],bin_edge[i+1]):
            B = torch.load('sampled_prob_0p2-1p7/new_sampled_prob_'+str(j+1)+'.pt')
            A[:,:,i] += B
    A=A/torch.sum(A)
    
    # Define the loss
    loss_fn = Loss_CNN_VAE(nb_beta,A,0.2)
    del(A,B)
    
    # Train the model
    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, rank, batch_size, loss_fn)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('num_training', type=int, help='Number of training samples')
    parser.add_argument('num_validation', type=int, help='Number of validation samples')
    parser.add_argument('batch_size', type=int, help='Input batch size on each device')
    parser.add_argument('n_poisson', type=int, help='Input number of poisson fluctuations')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.total_epochs, args.num_training, args.num_validation, args.batch_size, args.n_poisson), nprocs=world_size)
