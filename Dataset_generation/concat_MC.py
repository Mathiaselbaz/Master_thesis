import torch
import os

# Set the directory containing the .pt files
dir_path = "Events"

# Filter out files that are smaller than 300MB in size
pt_files = []
for f in os.listdir(dir_path):
    file_path = os.path.join(dir_path, f)
    file_size = os.path.getsize(file_path)
    if file_size <= 50000*1024*1024 and f.endswith(".pt"):
        pt_files.append(file_path)
# Load all the tensors from the .pt files and concatenate them along dimension 0
concatenated_tensor = torch.cat([torch.load(f) for f in pt_files], dim=0)[:,0:2].clone()

torch.save(concatenated_tensor,'Events/2023-05-09_MC_sample_10bins_400MeV-1GeV_Unif0-2_1M.pt')

