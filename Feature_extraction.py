# Standard library imports
import os
import csv
from tqdm import tqdm

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader 
from torchvision import transforms 

# Custom imports  
from Dataset import TileDataset
from Config import Config 

# Global variables
feature_dim = 512

# Feature extraction using encoder from AutoEncoder
def extract_features(input_folder_path: str, folder_name: str, n_samples: int, config: Config, device: torch.device, model: torch.nn.Module, actual_gpu_id=None):
    
    # Get GPU information for display
    device_display = f"GPU:{actual_gpu_id}" if actual_gpu_id is not None else str(device)
    
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor()
    ])
    
    dataset = TileDataset(folder_path=input_folder_path, config=config, transform=transform)
 
    if len(dataset) == 0:
        print(f"No images found in {input_folder_path}. Skipping.")
        return None, 0

    # Optimized DataLoader settings
    num_workers = 4 if device.type == 'cuda' else 2
    
    data_loader = DataLoader(dataset = dataset, batch_size = config.batch_size, shuffle = False, num_workers = num_workers,
        pin_memory = True if device.type == 'cuda' else False, persistent_workers = True if num_workers > 0 else False,
        prefetch_factor = 4 if num_workers > 0 else None)
    
    # Create temporary feature directory
    temp_feature_dir = os.path.join(config.output_path, 'temp_features', folder_name)
    os.makedirs(temp_feature_dir, exist_ok=True)
    
    # Determine where to save features
    if config.store_features:
        folder_feature_dir = os.path.join(config.feature_dir, folder_name)
        os.makedirs(folder_feature_dir, exist_ok=True)
        feature_file = os.path.join(folder_feature_dir, "features.csv")
    else:
        feature_file = os.path.join(temp_feature_dir, "features.csv")
    
    progress_desc = f"|{folder_name}| - |{device_display}| - |Features Extraction|"
    
    with open(feature_file, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        header = [f"Feature_{i}" for i in range(1, feature_dim + 1)] + ["File_Path"]
        writer.writerow(header)
        
        # Ensure model is on the correct device
        model = model.to(device)
        model.eval()
        
        # Use mixed precision for faster inference on GPU
        use_amp = device.type == 'cuda'
        
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc = progress_desc, unit = "batch", leave = True, ncols = 120,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            
            for batch_idx, (batch, file_paths) in enumerate(progress_bar):
                try:
                    batch = batch.to(device, non_blocking=True)
                    
                    # Use automatic mixed precision if GPU
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            latent = model.encoder(batch)
                            latent_gap = F.adaptive_avg_pool2d(latent, (1, 1))
                            latent_flattened = latent_gap.view(latent_gap.size(0), -1)
                    else:
                        latent = model.encoder(batch)
                        latent_gap = F.adaptive_avg_pool2d(latent, (1, 1))
                        latent_flattened = latent_gap.view(latent_gap.size(0), -1)
                    
                    # Move to CPU
                    latent_cpu = latent_flattened.cpu().numpy()
                    
                    # Batch write to CSV
                    rows = [list(feature_vector) + [path] for feature_vector, path in zip(latent_cpu, file_paths)]
                    writer.writerows(rows)
                    
                    # Update progress less frequently
                    if batch_idx % 10 == 0:
                        processed_samples = (batch_idx + 1) * config.batch_size
                        if device.type == 'cuda':
                            memory_info = f'{torch.cuda.memory_allocated(device) // 1024**2}MB'
                        else:
                            memory_info = 'N/A'
                        progress_bar.set_postfix({
                            'Samples': f'{min(processed_samples, len(dataset))}/{len(dataset)}',
                            'Memory': memory_info
                        })
                    
                    # Clear cache less frequently
                    if batch_idx % 100 == 0 and device.type == 'cuda':
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error processing batch {batch_idx} in {folder_name}: {str(e)}")
                    continue
       
    return feature_file, len(dataset), temp_feature_dir