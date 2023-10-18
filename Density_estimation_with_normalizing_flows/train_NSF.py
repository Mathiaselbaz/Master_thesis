# Import necessary libraries
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

# Import the custom Flows_model
from Flows_model import Flows_model

# Check if CUDA is available and set the device accordingly
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

# Function to set up Distributed Data Parallel (DDP) environment
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    



# Dataset class

class MC_Data(Dataset):
    def __init__(self, annotations_file, nb_train, start_data):
        self.dfi = pd.read_csv(annotations_file, names= ['pmu','Tmu','Enu'],delim_whitespace=True)
        self.dfi['pmu']=self.dfi['pmu']/10000
        self.dfi['Enu']=self.dfi['Enu']/10000
        self.dfi['Tmu']=np.arccos(self.dfi['Tmu'])/np.pi
        self.df2=torch.tensor(self.dfi[start_data:(start_data+nb_train)].to_numpy()).type(torch.float32)
        self.train=[]
        for i in range(nb_train):
            self.train.append(self.df2[i,:])
            
    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        return self.train[idx]
    
    def save_plot(self):
        Tmu=self.dfi['pmu'].to_numpy()
        Pmu=self.dfi['Tmu'].to_numpy()
        Enu=self.dfi['Enu'].to_numpy()
        sample=[]
        for i in range (100000):
            sample.append([Tmu[i],Pmu[i],Enu[i]])
        Tmu_MC=np.array(sample)[:,1]
        Pmu_MC=np.array(sample)[:,0]
        Enu_MC=np.array(sample)[:,2]
        fig, axs = plt.subplots(1,3,figsize=(9,3),sharex=False,sharey=False)
        fig.suptitle("MC events", fontsize=14)
        axs[0].hist2d(Tmu_MC,Pmu_MC,120,range=[[-0.05,1],[-0.05,1]])
        axs[0].set_xlabel('Tmu_fit (rad/pi)')
        axs[0].set_ylabel('Pmu_fit (GeV))')
        axs[1].hist2d(Tmu_MC,Enu_MC,120,range=[[-0.05,1],[-0.05,1]])
        axs[1].set_xlabel('Tmu_fit (rad/pi)')
        axs[1].set_ylabel('Enu_fit (GeV))')
        axs[2].hist2d(Pmu_MC,Enu_MC,120,range=[[-0.05,0.3],[-0.05,0.3]])
        axs[2].set_xlabel('Pmu_fit (GeV))')
        axs[2].set_ylabel('Enu_fit (GeV))')
        plt.tight_layout()
        plt.savefig('Plots/MC_for_training_histogram.png')
        plt.show()
        del(Tmu_MC,Pmu_MC,Enu_MC,Tmu,Pmu,Enu,sample)

        
        
# Class for training the model
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        valid_dataset: MC_Data,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        gpu_id: int,
        save_every: int,
        batch_size: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.valid_dataset = valid_dataset
        self.optimizer = optimizer
        self.scheduler=scheduler
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        self.batch_size=batch_size
        self.loss_tmp=0
        self.loss_train=[]
        self.loss_valid=[]

    def _run_batch(self, source):
        self.optimizer.zero_grad()
        self.loss_t= self.model.module.forward_kld(source)
        self.loss_v= self.model.module.forward_kld(self.valid_dataset.df2.to(self.gpu_id))
        self.loss_train=np.append(self.loss_train,self.loss_t.to('cpu').data.numpy())
        self.loss_valid=np.append(self.loss_valid,self.loss_v.to('cpu').data.numpy())
        self.loss_t.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = self.batch_size
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source in self.train_data:
            source = source.to(self.gpu_id)
            self._run_batch(source)

    def _save_checkpoint(self, epoch):
        self.model.module.eval()
        z=self.model.module.sample(10000)
        Tmu_fit=z[0].detach().cpu().numpy()[:,1]
        Pmu_fit=z[0].detach().cpu().numpy()[:,0]
        Enu_fit=z[0].detach().cpu().numpy()[:,2]

        fig, axs = plt.subplots(1,3,figsize=(9,3),sharex=False,sharey=False)
        fig.suptitle("Epoch "+str((epoch+1)*4*4), fontsize=14) 
        axs[0].hist2d(Tmu_fit,Pmu_fit,120,range=[[-0.05,1],[-0.05,1]])
        axs[0].set_xlabel('Tmu_fit (rad/pi)')
        axs[0].set_ylabel('Pmu_fit (GeV))')
        axs[1].hist2d(Tmu_fit,Enu_fit,120,range=[[-0.05,1],[-0.05,1]])
        axs[1].set_xlabel('Tmu_fit (rad/pi)')
        axs[1].set_ylabel('Enu_fit (GeV))')
        axs[2].hist2d(Pmu_fit,Enu_fit,120,range=[[-0.05,0.3],[-0.05,0.3]])
        axs[2].set_xlabel('Pmu_fit (GeV))')
        axs[2].set_ylabel('Enu_fit (GeV))')
        plt.tight_layout()
        plt.savefig('Plots/Epoch '+str((epoch+1)*4*4)+'.png')
        plt.show()
        del(z,Tmu_fit,Pmu_fit,Enu_fit)

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            self.scheduler.step()
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def plot_loss(save_every, total_epochs, train_hist, val_hist):
    epochs=np.arange(0, total_epochs, save_every, dtype=int)
    plt.figure(figsize=(10, 10))
    plt.plot(train_hist, label='loss')
    plt.plot(val_hist, label='loss_val')
    plt.xlabel('Epoch')
    plt.ylabel('log(Loss)')
    plt.legend()
    plt.savefig('Plots/Loss.png')
    plt.show()

    

def main(rank: int, world_size: int, save_every: int, total_epochs: int, num_training: int, num_validation: int, K: int, batch_size: int):
    # Set up distributed training
    ddp_setup(rank, world_size)
    train_dataset= MC_Data('mc_data.csv',num_training,1)
    valid_dataset= MC_Data('mc_data.csv',num_validation, num_training)
    train_dataset.save_plot()
    model=Flows_model(nf.distributions.DiagGaussian(3, trainable=False),K,128,5,False,3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.99, 0.999), weight_decay=1e-7)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    train_data = prepare_dataloader(train_dataset, batch_size)
    valid_data= prepare_dataloader(valid_dataset, batch_size)
    trainer = Trainer(model, train_data, valid_dataset, optimizer, scheduler, rank, save_every, batch_size)
    del(train_dataset, valid_dataset,optimizer,train_data,valid_data)
    trainer.train(total_epochs)
    destroy_process_group()
    plot_loss(save_every, total_epochs, trainer.loss_train, trainer.loss_valid)
    trainer.model.module.save('NSF_model.pt')



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('num_training', type=int, help='Number of training samples')
    parser.add_argument('num_validation', type=int, help='Number of validation samples')
    parser.add_argument('K', type=int, help='Number of flows in the model (default: 8)')
    parser.add_argument('batch_size', type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.num_training, args.num_validation, args.K, args.batch_size), nprocs=world_size)