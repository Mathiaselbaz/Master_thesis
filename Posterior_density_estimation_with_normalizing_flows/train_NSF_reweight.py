################ Imports packages and set up ddp ############################

import os
import sys
import nflows as nf
import numpy as np
from matplotlib import pyplot as plt
import torch
import pandas as pd
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

from nflows.flows.base import Flow
from nflows.flows.neural_spline.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressive
from Load_Dataset import Load_Dataset
from torchvision import models
from CNN_models import Trained_ResNet_4_Linear, Trained_ResNet_4_NSF
from torchvision.models import ResNet50_Weights

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
    



################ Creates training and validation datasets ############################

## Creation of Training and validation samples
class MC(Dataset):
    def __init__(self, data_dir, start, num_samples, beta_dir, n_poisson, num_beta = 10, num_samples_per_beta = 50000 , nb_bins= 200):
        self.beta = torch.load(beta_dir)[start*n_poisson:(start+num_samples)*n_poisson,:].clone()
        self.data = torch.load(data_dir)[start*num_samples_per_beta:(start+num_samples)*num_samples_per_beta,:].clone()
        self.num_beta =num_beta
        self.num_samples_per_beta = num_samples_per_beta
        self.n_poisson= n_poisson
        self.nb_bins =nb_bins

    def __len__(self):
        return self.data.size(0)//self.num_samples_per_beta

    def __getitem__(self, idx):
        data_idx = idx*self.num_samples_per_beta
        beta_idx=idx* self.n_poisson
        x = self.data[data_idx:data_idx+self.num_samples_per_beta,:].clone()
        nonzero_indices = torch.nonzero(torch.sum(x, dim=1))
        x_nonzero = x[nonzero_indices.squeeze()].clone()
        hist = torch.histogramdd(x_nonzero, bins=self.nb_bins, range=[0.02, 0.17, 0., 1.])[0].unsqueeze(0).repeat(3, 1, 1)
        beta=self.beta[beta_idx:beta_idx+self.n_poisson,:].clone()
        del(x_nonzero,x,nonzero_indices)      
        return hist,beta
    

# Modified RQ-NSF flow class (from nflows github)       
class AutoregressiveRationalQuadraticSpline(Flow):
    def __init__(
        self,
        num_input_channels,
        num_blocks,
        num_hidden_channels,
        num_bins=8,
        tail_bound=3,
        activation=torch.nn.ReLU,
        dropout_probability=0.0,
        permute_mask=False,
        init_identity=True,
    ):
        """Constructor

        Args:
          num_input_channels (int): Flow dimension
          num_blocks (int): Number of residual blocks of the parameter NN
          num_hidden_channels (int): Number of hidden units of the NN
          num_bins (int): Number of bins
          tail_bound (int): Bound of the spline tails
          activation (torch module): Activation function
          dropout_probability (float): Dropout probability of the NN
          permute_mask (bool): Flag, permutes the mask of the NN
          init_identity (bool): Flag, initialize transform as identity
        """
        super().__init__()

        self.mprqat = MaskedPiecewiseRationalQuadraticAutoregressive(
            features=num_input_channels,
            hidden_features=num_hidden_channels,
            context_features=10,
            num_bins=num_bins,
            tails="linear",
            tail_bound=tail_bound,
            num_blocks=num_blocks,
            use_residual_blocks=True,
            random_mask=False,
            permute_mask=permute_mask,
            activation=activation(),
            dropout_probability=dropout_probability,
            use_batch_norm=False,
            init_identity=init_identity,
        )

    def forward(self, z,y):
        z, log_det = self.mprqat.inverse(z,y)
        return z, log_det

    def inverse(self, z, y):
        z, log_det = self.mprqat(z,y)
        return z, log_det

# Function to define the model    
def Flows_model(q0,K,hidden_units,hidden_layers,latent_size):
    flows = []
    for i in range(K):
        flows += [AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units)]
        flows += [nf.flows.LULinearPermute(latent_size)]
    # Construct flow model
    net=Trained_ResNet_4_NSF(10)
    input_data = torch.randn((1,3,200,200))
    _ = net(input_data)
    model = nf.ClassCondFlow(q0=q0, flows=flows, net=net)
    return model
  


# Class trainer        
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        gpu_id: int,
        batch_size: int
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler=scheduler
        self.model = DDP(model, device_ids=[gpu_id])
        self.batch_size=batch_size
        self.loss_train=[]
        self.loss_valid=[]
        self.n_poisson=1000
        self.n_beta=3
        self.run=str(1)
        self.start_flow=10

    def _run_batch(self, source,betas):
        loss_tmp=0
        self.optimizer.zero_grad()   
        loss= self.model.module.forward_kld(betas.to(self.gpu_id),source.to(self.gpu_id))
        loss_t=loss/self.n_poisson
        loss_t.backward()
        self.optimizer.step()
        self.loss_train.append(loss_t.item())
        del(betas,source)
        
        
        # Plot losses
        plt.figure()
        fig,axs = plt.subplots(1, 2, figsize=(10, 5),sharex=False,sharey=False)
        ax1 =axs[0].plot(self.loss_train)
        axs[0].set_xlabel('Batch number')
        axs[0].set_ylabel('Training loss')
        plt.tight_layout()
        ax2 =axs[1].plot(self.loss_valid)
        axs[1].set_xlabel('Batch number')
        axs[1].set_ylabel('Validation loss')
        plt.tight_layout()
        plt.savefig('Plots/Loss.png')
        plt.show()

        torch.cuda.empty_cache()
        
       

    def _run_batch_val(self, source,betas):    
        loss_tmp=0  
        loss= self.model.module.forward_kld(betas.to(self.gpu_id),source.to(self.gpu_id))
        loss_v=loss/self.n_poisson
        self.loss_valid.append(loss_v.item())
       
        
        # Long code to make all the plots during test (until line 288)
        plt.figure()
        fig,axs = plt.subplots(1, 2, figsize=(10, 5),sharex=False,sharey=False)
        ax1 =axs[0].plot(self.loss_train)
        axs[0].set_xlabel('Batch number')
        axs[0].set_ylabel('Training loss')
        plt.tight_layout()
        ax2 =axs[1].plot(self.loss_valid)
        axs[1].set_xlabel('Batch number')
        axs[1].set_ylabel('Validation loss')
        plt.tight_layout()
        plt.savefig('Plots/Loss_'+self.run+'.png')
        plt.show()
        
        self.model.module.eval()
        z=self.model.module.sample(self.n_poisson,source[0,:,:,:].clone().unsqueeze(0).to(source.device))
        Betas_pred=z[0].squeeze(0).detach().cpu().numpy()
        del(z,source)
        Betas_true=betas[0,:,:].clone().detach().cpu().numpy()
        

        # Create subplots for Betas_pred
        plt.figure()
        fig, axs = plt.subplots(3, 3, figsize=(10, 10), sharex=False, sharey=False)


        for i in range(self.n_beta):
            for j in range(self.n_beta):
                if i == j:
                    # Plot 1D histogram for Betas_pred[j]
                    axs[i, j].hist(Betas_pred[:, j], bins=50, range=[np.min(Betas_true[:, j])-0.1, np.max(Betas_true[:, j])+0.1], color='blue')
                    axs[i, j].set_xlabel('Bin ' + str(j))
                    axs[i, j].set_ylabel('Frequency')
                    axs[i, j].set_xlim([np.min(Betas_true[:, j])-0.1, np.max(Betas_true[:, j])+0.1])

                    # Calculate and add the standard deviation (sigma) and mean as text on the plot
                    std_label = r'$\sigma$ = {:.3f}'.format(np.std(Betas_pred[:, j]))
                    mean_label = 'Mean = {:.3f}'.format(np.mean(Betas_pred[:, j]))

                    # Adjust the position of text in the top-right corner
                    axs[i, j].text(0.95, 0.80, std_label, transform=axs[i, j].transAxes, ha='right', va='top', fontsize='small')
                    axs[i, j].text(0.95, 0.90, mean_label, transform=axs[i, j].transAxes, ha='right', va='top', fontsize='small')
                else:
                    # Plot 2D histogram for Betas_pred[:, i] and Betas_pred[:, j]
                    axs[i, j].hist2d(Betas_pred[:, i], Betas_pred[:, j], 50, range=[[np.min(Betas_true[:, i])-0.1, np.max(Betas_true[:, i])+0.1],[np.min(Betas_true[:, j])-0.1, np.max(Betas_true[:, j])+0.1]])
                    axs[i, j].set_xlabel('Bin ' + str(i))
                    axs[i, j].set_ylabel('Bin ' + str(j))
                    axs[i, j].set_xlim([np.min(Betas_true[:, i])-0.1, np.max(Betas_true[:, i])+0.1])
                    axs[i, j].set_ylim([np.min(Betas_true[:, j])-0.1, np.max(Betas_true[:, j])+0.1])

        # Save the plot
        plt.tight_layout()
        plt.savefig('Plots/Beta_pred_'+self.run+'.png')
        plt.show()

        # Create subplots for Betas_true
        plt.figure()
        fig, axs = plt.subplots(3, 3, figsize=(10, 10), sharex=False, sharey=False)

        for i in range(self.n_beta):
            for j in range(self.n_beta):
                if i == j:
                    # Plot 1D histogram for Betas_true[j]
                    axs[i, j].hist(Betas_true[:, j], bins=50, range=[np.min(Betas_true[:, j])-0.1, np.max(Betas_true[:, j])+0.1], color='blue')
                    axs[i, j].set_xlabel('Bin ' + str(j))
                    axs[i, j].set_ylabel('Frequency')
                    axs[i, j].set_xlim([np.min(Betas_true[:, j])-0.1, np.max(Betas_true[:, j])+0.1])

                    # Calculate and add the standard deviation (sigma) and mean as text on the plot
                    std_label = r'$\sigma$ = {:.3f}'.format(np.std(Betas_true[:, j]))
                    mean_label = r'$\mu$ = {:.3f}'.format(np.mean(Betas_true[:, j]))

                    # Adjust the position of text in the top-right corner
                    axs[i, j].text(0.95, 0.80, std_label, transform=axs[i, j].transAxes, ha='right', va='top', fontsize='small')
                    axs[i, j].text(0.95, 0.90, mean_label, transform=axs[i, j].transAxes, ha='right', va='top', fontsize='small')
                else:
                    # Plot 2D histogram for Betas_true[:, i] and Betas_true[:, j]
                    axs[i, j].hist2d(Betas_true[:, i], Betas_true[:, j], 50, range=[[np.min(Betas_true[:, i])-0.1, np.max(Betas_true[:, i])+0.1],[np.min(Betas_true[:, j])-0.1, np.max(Betas_true[:, j])+0.1]])
                    axs[i, j].set_xlabel('Bin ' + str(i))
                    axs[i, j].set_ylabel('Bin ' + str(j))
                    axs[i, j].set_xlim([np.min(Betas_true[:, i])-0.1, np.max(Betas_true[:, i])+0.1])
                    axs[i, j].set_ylim([np.min(Betas_true[:, j])-0.1, np.max(Betas_true[:, j])+0.1])

        # Save the plot
        plt.tight_layout()
        plt.savefig('Plots/Beta_true_'+self.run+'.png')
        
        plt.show()
        
        # Add torch.cuda.empty_cache()
        torch.cuda.empty_cache()


    def _run_epoch(self, epoch):
        b_sz = self.batch_size
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_loader)}")
        self.train_loader.sampler.set_epoch(epoch)
        self.val_loader.sampler.set_epoch(epoch)
        for source,beta in self.train_loader:
            source = source.to(self.gpu_id)
            beta=beta.to(self.gpu_id)
            self._run_batch(source,beta)
        for val,beta_val in self.val_loader:
            val = val.to(self.gpu_id)
            beta_val=beta_val.to(self.gpu_id)
            self._run_batch_val(val,beta_val)
            

    def train(self, max_epochs: int):
        self.model.module.eval()
        for param in self.model.module.flows.parameters():
            param.requires_grad = False
        for param in self.model.module.net.parameters():
            param.requires_grad = False
            
        for epoch in range(max_epochs):
            if epoch == self.start_flow:
                for param in self.model.module.flows.parameters():
                    param.requires_grad = True
                for param in self.model.module.net.fc1.parameters():
                    param.requires_grad = True
                                    
            self._run_epoch(epoch)
            self.scheduler.step()


    

def main(rank: int, world_size: int, total_epochs: int, num_training: int, num_validation: int, K: int, batch_size: int):
    ddp_setup(rank, world_size)
    
    #Create the datasets
    data_dir = 'Events/2023-05-18_MC_sample_3bins_200MeV-1p7GeV_Unif0-2_1128M_equifilled_rowbeta_100k.pt'
    beta_dir = 'Events/2023-05-18_MC_sample_3bins_200MeV-1p7GeV_Unif0-2_1128M_equifilled_rowbeta_100k_beta_1000_island1-2v2.pt'
    train_dataset = MC(data_dir = data_dir, start = 1, num_samples = num_training, beta_dir=beta_dir, n_poisson=1000, num_beta = 3, num_samples_per_beta = 100000, nb_bins= 200)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False,pin_memory=True,drop_last=False, sampler=DistributedSampler(train_dataset))
    val_dataset = MC(data_dir = data_dir, start = num_training+1, num_samples = num_validation, beta_dir=beta_dir, n_poisson=1000, num_beta = 3, num_samples_per_beta = 100000, nb_bins= 200)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, pin_memory=True,drop_last=False,sampler=DistributedSampler(val_dataset))
    del(train_dataset,val_dataset)

    # Create the linear flow and its encoder network
    CNN=Trained_ResNet_4_Linear(3)
    input_data = torch.randn((batch_size,3,200,200))
    _ = CNN(input_data)
    del(input_data)
    q0=nf.distributions.MyGaussianFlow(CNN)
    
    # Create the ANF model
    model=Flows_model(q0,K,128,3,3)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.95, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs, 1e-7, last_epoch=- 1, verbose=True)
    
    #Train and save the model
    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, rank, batch_size)
    trainer.train(total_epochs)
    destroy_process_group()
    trainer.model.module.save('Plots/Model_nsf_reweight.pt')



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('num_training', type=int, help='Number of training samples')
    parser.add_argument('num_validation', type=int, help='Number of validation samples')
    parser.add_argument('K', type=int, help='Number of flows in the model (default: 8)')
    parser.add_argument('batch_size', type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.total_epochs, args.num_training, args.num_validation, args.K, args.batch_size), nprocs=world_size)