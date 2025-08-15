# ====================================
# Feature_extraction.py
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

def extract_features(target_folder_path: str, subfolder_name: str, n_samples: int, config: Config, device: torch.device, model: torch.nn.Module): 
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Pre-calculate and display the expected number of clusters  
    print(f"--- Processing : {subfolder_name} ---")  
    print("-----------------------------------------------------")
    print(f"Number of samples available: {n_samples} in {subfolder_name}") 
    
    dataset = TileDataset(folder_path=target_folder_path, config=config, transform=transform)
    
    if len(dataset) == 0:
        print(f"No images found in {target_folder_path}. Skipping.")
        return None, 0
        
    # Reduce num_workers to prevent multiprocessing issues
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,  # Changed from 2 to 0 to avoid multiprocessing issues
        pin_memory=False if device.type == 'cpu' else True  # Only pin memory for GPU
    )
    
    # Create temporary feature directory for processing (always needed for clustering)
    temp_feature_dir = os.path.join(config.output_path, 'temp_features', subfolder_name)
    os.makedirs(temp_feature_dir, exist_ok=True)
    
    # Determine where to save features based on store_features flag
    if config.store_features:
        subfolder_feature_dir = os.path.join(config.feature_dir, subfolder_name)
        os.makedirs(subfolder_feature_dir, exist_ok=True)
        feature_file = os.path.join(subfolder_feature_dir, "features.csv")
    else:
        feature_file = os.path.join(temp_feature_dir, "features.csv")
    
    feature_dim = 512
    
    with open(feature_file, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        header = [f"Feature_{i}" for i in range(1, feature_dim + 1)] + ["File_Path"]
        writer.writerow(header)
        
        with torch.no_grad():
            for batch_idx, (batch, file_paths) in enumerate(tqdm(data_loader, desc=f"Extracting features for {subfolder_name}")):
                try:
                    batch = batch.to(device)
                    latent = model.encoder(batch)
                    latent_gap = F.adaptive_avg_pool2d(latent, (1, 1))
                    latent_flattened = latent_gap.view(latent_gap.size(0), -1).cpu().numpy()
                    
                    for feature_vector, path in zip(latent_flattened, file_paths):
                        writer.writerow(list(feature_vector) + [path])
                    
                    # Clear GPU cache periodically to prevent memory issues
                    if batch_idx % 100 == 0 and device.type == 'cuda':
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {str(e)}")
                    continue
    
    if config.store_features:
        print(f"Features extracted and saved to {feature_file}")
    else:
        print(f"Features extracted (temporary file for processing)")
    
    return feature_file, len(dataset), temp_feature_dir