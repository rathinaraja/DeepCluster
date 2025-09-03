# ====================================
# Feature_extraction.py (Enhanced with GPU info)
# ====================================
import os
import csv
from tqdm import tqdm 

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision import transforms

from Dataset import TileDataset
from Config import Config 

feature_dim = 512

def extract_features(input_folder_path: str, folder_name: str, n_samples: int, config: Config, device: torch.device, model: torch.nn.Module, actual_gpu_id=None): 
    
    # Get GPU information for display
    device_display = f"GPU:{actual_gpu_id}" if actual_gpu_id is not None else str(device)
    
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor() ])
    
    dataset = TileDataset(folder_path=input_folder_path, config=config, transform=transform)
 
    if len(dataset) == 0:
        print(f"No images found in {input_folder_path}. Skipping.")
        return None, 0

    # Optimize DataLoader for GPU usage
    data_loader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True if device.type == 'cuda' else False )
    
    # Create temporary feature directory for processing
    temp_feature_dir = os.path.join(config.output_path, 'temp_features', folder_name)
    os.makedirs(temp_feature_dir, exist_ok=True)
    
    # Determine where to save features based on store_features flag
    if config.store_features:
        folder_feature_dir = os.path.join(config.feature_dir, folder_name)
        os.makedirs(folder_feature_dir, exist_ok=True)
        feature_file = os.path.join(folder_feature_dir, "features.csv")
    else:
        feature_file = os.path.join(temp_feature_dir, "features.csv")
    
    # Enhanced progress bar description with actual GPU info and folder
    progress_desc = f"|{folder_name}| - |{device_display}| - Features Extraction"
    
    with open(feature_file, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        header = [f"Feature_{i}" for i in range(1, feature_dim + 1)] + ["File_Path"]
        writer.writerow(header)
        
        # Ensure model is on the correct device
        model = model.to(device)
        model.eval()  # Set to evaluation mode for feature extraction
        
        with torch.no_grad():
            # Enhanced progress bar
            progress_bar = tqdm(data_loader, desc = progress_desc, unit = "batch", leave = True, ncols = 120,  # Wider progress bar to show more info
                bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
            
            for batch_idx, (batch, file_paths) in enumerate(progress_bar):
                try:
                    # Move batch to device with non_blocking for better GPU utilization
                    batch = batch.to(device, non_blocking=True)
                    
                    # Extract features
                    latent = model.encoder(batch)
                    latent_gap = F.adaptive_avg_pool2d(latent, (1, 1))
                    latent_flattened = latent_gap.view(latent_gap.size(0), -1).cpu().numpy()
                    
                    # Write features to CSV
                    for feature_vector, path in zip(latent_flattened, file_paths):
                        writer.writerow(list(feature_vector) + [path])
                    
                    # Update progress bar with additional info
                    if batch_idx % 10 == 0:  # Update every 10 batches
                        processed_samples = (batch_idx + 1) * config.batch_size
                        memory_info = f'{torch.cuda.memory_allocated(device) // 1024**2}MB' if device.type == 'cuda' else 'N/A'
                        progress_bar.set_postfix({ 'Samples': f'{min(processed_samples, len(dataset))}/{len(dataset)}', 'Memory': memory_info })
                    
                    # Clear GPU cache periodically to prevent memory issues
                    if batch_idx % 50 == 0 and device.type == 'cuda':
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error processing batch {batch_idx} in {folder_name}: {str(e)}")
                    continue
       
    return feature_file, len(dataset), temp_feature_dir