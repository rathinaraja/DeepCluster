import os
import math
import csv
import warnings
import argparse
import logging
import sys
from pathlib import Path
from typing import List
from threading import Lock
import concurrent.futures

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN

import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torchvision import transforms

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

import Auto_encoder 

# Common image extensions used throughout the module
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff') 

# ====================================
# Config 
# ==================================== 
class Config:
    def __init__(self, input_path: str, input_folders: str, sub_folders: str, output_path: str, process_all: str, batch_size: int, 
                 dim_reduce: int, num_distance_groups: int,  sample_percentage: float,  store_features: bool, store_clusters: bool, 
                 store_plots: bool, store_samples: bool, store_samples_group_wise: bool):
        self.input_path = input_path
        self.input_folders = input_folders
        self.sub_folders = sub_folders 
        self.output_path = output_path
        self.process_all = process_all
        self.batch_size = batch_size
        self.dim_reduce = dim_reduce
        self.num_distance_groups = num_distance_groups
        self.sample_percentage = sample_percentage
        self.store_features = store_features
        self.store_clusters = store_clusters
        self.store_plots = store_plots
        self.store_samples = store_samples
        self.store_samples_group_wise = store_samples_group_wise  
        
        # Check the input parameters
        print("\n-----------------------------------------------------")
        print("Input/output paths...")
        print("-----------------------------------------------------")
        print("Input_path\t:",self.input_path)
        print("Output_path\t:",self.output_path) 
        
        if self.sub_folders: 
            print("Sub_folders\t:", self.sub_folders) 
        else:
            print("Sub_folders\t: all") 

        if self.process_all:
            print("Processing\t: all images in all folders and subfolders") 
            
        if self.store_features:
            self.feature_dir = os.path.join(output_path, 'features')
            os.makedirs(self.feature_dir, exist_ok=True)
            print("Feature_path\t:",self.feature_dir)
        else:
            print("Feature_path\t: not created (temporary processing)")
            
        if self.store_clusters:
            self.cluster_dir = os.path.join(output_path, 'clusters')
            os.makedirs(self.cluster_dir, exist_ok=True)
            print("Cluster_path\t:",self.cluster_dir)
        else:
            print("Cluster_path\t: not created")

        if self.store_plots:
            self.plot_dir = os.path.join(output_path, 'plots')
            os.makedirs(self.plot_dir, exist_ok=True)
            print("t-sne_plot_path\t:",self.plot_dir) 
        else:
            print("t-sne_plot_path\t: not created") 

        if self.store_samples:            
            if self.store_samples_group_wise:
                print("Samples_path\t:",self.output_path,"/samples") 
                print("Samples_stored\t: group folder wise") 
            else:
                print("Samples_path\t:",self.output_path,"/samples") 
                print("Samples_stored\t: cluster folder wise") 
        else:
            print("Samples_path\t: not created")
             
        print("-----------------------------------------------------")
        print("Configuration parameters...")
        print("-----------------------------------------------------")
        print("Batch_size\t\t:",self.batch_size)
        print("Sample_percentage\t:",self.sample_percentage)
        print("Num_distance_groups\t:",self.num_distance_groups)
        print("Store_features\t\t:",self.store_features)
        print("-----------------------------------------------------")
        print("Algorithms used...")
        print("-----------------------------------------------------")
        print("Feature extraction\t: AutoEncoder-based encoder")
        print("Feature_size\t\t: 512 ")
        print(f"Feature_reduction\t: PCA (512 -> {self.dim_reduce})")
        print("Clustering_algom\t: K-means")
        print(f"Feature_visualization\t: t-SNE ({self.dim_reduce} -> 2)")   

# ====================================
# Dataset  
# ==================================== 
def get_sub_folders_from_config(folder_path: str, config: Config):
    """
    input : path to a input folder, /path/WSI1
    output: list of paths of subfolders /path/WSI1/Sub1, /path/WSI1/sub2, etc.  
    """
    target_sub_folders = []
    
    if config.sub_folders is None:
        # Processes all subfolders and flat images in the input folder 
        target_sub_folders.append(folder_path)
        
    else:
        # Parse comma-separated subfolder names
        sub_folder_names = [name.strip() for name in config.sub_folders.split(',')]
        for name in sub_folder_names:
            target_sub_folder_path = os.path.join(folder_path, name)
            target_sub_folders.append(target_sub_folder_path)
    
    return target_sub_folders

def collect_image_files(target_folders):
    """
    input: list of sub-folder paths for a WSI
    output: list of image paths from the respective subfolders of a WSI
    """
    image_files = []
    
    for folder in target_folders:
        if not os.path.isdir(folder):
            print(f"Warning: Folder '{folder}' does not exist. Skipping.")
            continue
        
        # Collect all image files recursively
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(IMAGE_EXTENSIONS):
                    image_files.append(os.path.join(root, file))
    
    return image_files

def count_images_in_folder(folder_path, config):
    """
    Count the number of image files in a folder and its subfolders. 
    """
    # Use the same logic as TileDataset to ensure consistency
    target_folders = get_sub_folders_from_config(folder_path, config)  
    image_files = collect_image_files(target_folders) 
    return len(image_files)

class TileDataset(Dataset):
    def __init__(self, folder_path: str, config: Config, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        
        # Use centralized functions to avoid code duplication
        sub_folders = get_sub_folders_from_config(folder_path, config)
        self.image_files = collect_image_files(sub_folders)
        
    def __len__(self):  
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, img_path
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a dummy tensor and path instead of recursing
            if self.transform:
                dummy_img = self.transform(Image.new('RGB', (256, 256), color=(0, 0, 0)))
            else:
                dummy_img = Image.new('RGB', (256, 256), color=(0, 0, 0))
            return dummy_img, img_path

# ====================================
# Sampling
# ====================================
def sample_clustered_images(subfolder_name: str, n_clusters: int, config: Config, temp_feature_dir: str):     
    try:
        # Determine where to read files from based on store_features flag
        if config.store_features:
            subfolder_feature_dir = os.path.join(config.feature_dir, subfolder_name)
            assignment_file = os.path.join(subfolder_feature_dir, "cluster_assignments.csv")
            feature_file = os.path.join(subfolder_feature_dir, "features.csv")
        else:
            assignment_file = os.path.join(temp_feature_dir, "cluster_assignments.csv")
            feature_file = os.path.join(temp_feature_dir, "features.csv")
        
        if not os.path.exists(assignment_file) or not os.path.exists(feature_file):
            print(f"Missing required files for sample selection in {subfolder_name}")
            return
        
        # Load assignments and features
        assignments = pd.read_csv(assignment_file)
        features_df = pd.read_csv(feature_file)
        
        # Prepare output directory
        samples_base = os.path.join(config.output_path, 'samples', subfolder_name)
        os.makedirs(samples_base, exist_ok=True)
        
        # Process each cluster
        cluster_ids = assignments['Cluster'].unique()
        
        for cluster_id in tqdm(cluster_ids, desc=f"Sampling clusters for {subfolder_name}"):
            # Get files and features for this cluster
            cluster_mask = assignments['Cluster'] == cluster_id
            cluster_files = assignments.loc[cluster_mask, 'File_Path'].values
            
            # Skip if cluster is empty
            if len(cluster_files) == 0:
                continue
            
            # Create output directory for this cluster
            cluster_sample_dir = os.path.join(samples_base, f'Cluster_{cluster_id}')
            os.makedirs(cluster_sample_dir, exist_ok=True)
            
            # Match files with their features
            cluster_features = []
            cluster_file_paths = []
            
            for file_path in cluster_files:
                feature_row = features_df[features_df['File_Path'] == file_path].drop('File_Path', axis=1).values
                if len(feature_row) > 0:
                    cluster_features.append(feature_row[0])
                    cluster_file_paths.append(file_path)
            
            if not cluster_features:
                continue
            
            # Convert to numpy array for calculations
            cluster_features = np.array(cluster_features)
            
            # Calculate centroid
            centroid = np.mean(cluster_features, axis=0)
            
            # Calculate distances from centroid to each sample
            distances = []
            for i, feature in enumerate(cluster_features):
                # Use Euclidean distance
                dist = np.linalg.norm(feature - centroid)
                distances.append((cluster_file_paths[i], dist))
            
            # Normalize distances to [0, 1]
            if len(distances) > 1:  # Only normalize if more than one sample
                max_dist = max(d[1] for d in distances)
                min_dist = min(d[1] for d in distances)
                
                # Handle case where all points are the same distance
                if max_dist == min_dist:
                    normalized_distances = [(file_path, 0.0) for file_path, _ in distances]
                else:
                    normalized_distances = [
                        (file_path, (dist - min_dist) / (max_dist - min_dist))
                        for file_path, dist in distances
                    ]
            else:
                # If only one sample, its normalized distance is 0
                normalized_distances = [(distances[0][0], 0.0)]
            
            # Implementation of equal-frequency binning
            sorted_distances = sorted(normalized_distances, key=lambda x: x[1])
            total_samples = len(sorted_distances)
            
            # Calculate base samples per group and remainder
            samples_per_group = total_samples // config.num_distance_groups
            remainder = total_samples % config.num_distance_groups
            
            # Initialize empty groups
            distance_groups = [[] for _ in range(config.num_distance_groups)]
            current_idx = 0
            
            # Distribute samples using equal-frequency binning
            for group_idx in range(config.num_distance_groups):
                # Add extra sample to earlier groups if there's a remainder
                extra = 1 if group_idx < remainder else 0
                group_size = samples_per_group + extra
                
                # Last group gets any remaining samples
                if group_idx == config.num_distance_groups - 1:
                    group_size = total_samples - current_idx
                
                # Add samples to this group
                for i in range(group_size):
                    if current_idx < total_samples:
                        file_path, _ = sorted_distances[current_idx]
                        distance_groups[group_idx].append(file_path)
                        current_idx += 1
            
            # Sample from each distance group 
            for group_idx, group_files in enumerate(distance_groups):
                if not group_files:
                    continue
                    
                # Calculate number of samples to select
                num_samples = max(1, int(len(group_files) * config.sample_percentage))                 
                # if int(len(group_files) * config.sample_percentage) < 1:
                #     continue
               
                # Randomly select samples
                selected_files = np.random.choice(
                    group_files, 
                    size=min(num_samples, len(group_files)), 
                    replace=False
                )

                # Switch to control samples to store group-wise or flat for each cluster
                if config.store_samples_group_wise:
                    # Create group directory
                    group_dir = os.path.join(cluster_sample_dir, f'Group_{group_idx}')
                    os.makedirs(group_dir, exist_ok=True)
                    
                    #Copy selected files to output directory
                    for file_path in selected_files:
                        try:
                            img = Image.open(file_path)
                            img.save(os.path.join(group_dir, os.path.basename(file_path)))
                        except Exception as e:
                            print(f"Error processing image {file_path}: {e}") 
                else:
                    # Copy selected files cluster folder wise
                    for file_path in selected_files:
                        try:
                            img = Image.open(file_path)
                            img.save(os.path.join(cluster_sample_dir, os.path.basename(file_path)))
                        except Exception as e:
                            print(f"Error processing image {file_path}: {e}") 
    
    except Exception as e:
        print(f"Error in sample_clustered_images: {str(e)}")
        raise

# ====================================
# Clustering
# ====================================
def visualize_clusters(features_2d, clusters, file_paths, output_prefix: str, palette):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    # Use matplotlib's Agg backend for headless environments
    plt.switch_backend('Agg')
    
    # Plot with legend
    plt.figure(figsize=(12, 8))
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[palette[cluster]],
            label=f'Cluster {cluster}',
            alpha=0.6
        )
        
        # Draw convex hull
        if np.sum(mask) >= 3:  # Need at least 3 points for a hull
            try:
                hull = ConvexHull(features_2d[mask])
                for simplex in hull.simplices:
                    plt.plot(features_2d[mask][simplex, 0], 
                            features_2d[mask][simplex, 1], 
                            c=palette[cluster])
            except Exception as e:
                print(f"Could not create convex hull for cluster {cluster}: {e}")
    
    plt.title("t-SNE Visualization with K-Means Clusters")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_with_legend.png", dpi=400, bbox_inches='tight') 
    plt.close()
    plt.clf()  # Clear figure memory
    
    # Plot with cluster numbers
    plt.figure(figsize=(16, 11))
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[palette[cluster]],
            alpha=0.6
        )
        
        # Add cluster number in center
        if np.sum(mask) > 0:  # Only if cluster has points
            center = features_2d[mask].mean(axis=0)
            plt.text(center[0], center[1], str(cluster), 
                    horizontalalignment='center', 
                    verticalalignment='center',
                    fontweight='bold')
        
        # Draw convex hull
        if np.sum(mask) >= 3:
            try:
                hull = ConvexHull(features_2d[mask])
                for simplex in hull.simplices:
                    plt.plot(features_2d[mask][simplex, 0], 
                            features_2d[mask][simplex, 1], 
                            c=palette[cluster])
            except Exception as e:
                print(f"Could not create convex hull for cluster {cluster}: {e}")
    
    plt.title("t-SNE Visualization with Cluster Numbers")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_with_numbers.png", dpi=400, bbox_inches='tight') 
    plt.close()
    plt.clf()  # Clear figure memory 

def calc_num_clusters(n_samples):
    # Number of clusters adjustment (less cluster more samples per cluster) 0 - 1 (more cluster less samples per cluster)
    a = 1   
    n_clusters = math.ceil(a * math.sqrt(n_samples))
    return n_clusters
    
def process_clusters(subfolder_name: str, feature_file: str, n_samples: int, config: Config, temp_feature_dir: str): 
    try:
        data = pd.read_csv(feature_file)
        features = data.drop('File_Path', axis=1).values
        file_paths = data['File_Path'].values 

        if n_samples < n_samples:
            n_clusters = 1
        else:
            n_clusters = calc_num_clusters(n_samples)
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Calculate safe perplexity value (must be less than n_samples)
        perplexity = min(30, max(5, min(n_samples - 1, n_samples // 10))) 
        n_components = config.dim_reduce
        print(f"\nApplying PCA to reduce features from 512 to {n_components}...")
        pca = PCA(n_components)
        features_reduced_pca = pca.fit_transform(features_scaled)
       
        print(f"\nApplying K-Means with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_reduced_pca) 

        if config.store_plots:         
            # Create subfolder-specific plot directory
            plot_dir = os.path.join(config.plot_dir, subfolder_name)
            os.makedirs(plot_dir, exist_ok=True)
    
            print(f"\nApplying t-SNE to visualize the clusters ({n_components} -> 2)...\n")
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1500, learning_rate='auto', init="pca", n_jobs=1)
            features_2d_tsne = tsne.fit_transform(features_reduced_pca)
    
            palette = sns.color_palette("husl", n_colors=n_clusters)
            visualize_clusters(features_2d_tsne, clusters, file_paths, os.path.join(plot_dir, "tsne"), palette)
        
        # Save cluster assignments
        assignments = pd.DataFrame({
            'File_Path': file_paths,
            'Cluster': clusters
        })
        
        # Save cluster assignments based on store_features flag
        if config.store_features:
            feature_dir = os.path.dirname(feature_file)
            assignments.to_csv(os.path.join(feature_dir, "cluster_assignments.csv"), index=False)
        else:
            # Save to temporary directory for sampling process
            assignments.to_csv(os.path.join(temp_feature_dir, "cluster_assignments.csv"), index=False) 
            
        if config.store_clusters:
            # Create subfolder-specific cluster directory
            cluster_base = os.path.join(config.cluster_dir, subfolder_name)
            os.makedirs(cluster_base, exist_ok=True)
    
            for cluster in tqdm(range(n_clusters), desc=f"Creating cluster folders for {subfolder_name}"):
                cluster_dir = os.path.join(cluster_base, f'Cluster_{cluster}')
                os.makedirs(cluster_dir, exist_ok=True)
                
                cluster_files = file_paths[clusters == cluster] 
                for file_path in cluster_files:
                    try:  
                        img = Image.open(file_path)
                        img.save(os.path.join(cluster_dir, os.path.basename(file_path)))
                    except Exception as e:
                        print(f"Error processing image {file_path}: {e}")
            
            # Print the actual samples per cluster distribution 
            cluster_counts = np.bincount(clusters)
            avg_samples_per_cluster = n_samples / n_clusters if n_clusters > 0 else 0
            # print(f"Actual distribution: Avg {avg_samples_per_cluster:.1f} samples/cluster")
            # print(f"Min: {cluster_counts.min()} samples, Max: {cluster_counts.max()} samples\n")
        
        return assignments, n_clusters
        
    except Exception as e:
        print(f"Error in process_clusters: {str(e)}")
        raise

# ====================================
# Feature_extraction 
# ====================================
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

# ====================================
# Processing_parallel
# ====================================
def cleanup_temporary_files(temp_feature_dir: str, config: Config):
    """Clean up temporary feature files if store_features is False"""
    if not config.store_features and os.path.exists(temp_feature_dir):
        try:
            import shutil
            shutil.rmtree(temp_feature_dir)
            print(f"Cleaned up temporary feature files from {temp_feature_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up temporary files from {temp_feature_dir}: {e}")

def process_input_folder(input_folder_path: str, config: Config, device: torch.device, model: torch.nn.Module):
    subfolder_name = os.path.basename(input_folder_path)
    temp_feature_dir = None
    
    try:
        # First count the number of images to estimate the number of clusters
        n_samples = count_images_in_folder(input_folder_path, config)

        if n_samples == 0:
            print(f"No images found in subfolder: {subfolder_name}. Skipping.")
            print(f"----------------------------------------------------\n")
            return False
        # print(input_folder_path)
        # print(subfolder_name)
  
        # Extract features
        result = extract_features(input_folder_path, subfolder_name, n_samples, config, device, model)
        if len(result) == 3:
            feature_file, actual_samples, temp_feature_dir = result
        else:
            feature_file, actual_samples = result
            temp_feature_dir = None
        
        if feature_file is not None:
            # Process clusters
            assignments, n_clusters = process_clusters(subfolder_name, feature_file, actual_samples, config, temp_feature_dir or "")
            
            # Sample representative images
            if config.store_samples:
                sample_clustered_images(subfolder_name, n_clusters, config, temp_feature_dir or "")
            
            # Clean up temporary files if needed
            if temp_feature_dir:
                cleanup_temporary_files(temp_feature_dir, config)
            
            print(f"\nProcessing completed for WSI : {subfolder_name}")
            print(f"-----------------------------------------------------")
            return True
        else:
            print(f"Feature extraction failed for subfolder: {subfolder_name}. Skipping.")
            return False
    except Exception as e:
        print(f"\nError processing WSI {subfolder_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Clean up temporary files in case of error
        if temp_feature_dir:
            cleanup_temporary_files(temp_feature_dir, config)        
        return False
            
def get_input_folders_list(input_path, config: Config): 
    """
    Input: Input path containing a set of folders (WSI1, WSI2,...,etc) corresponding to WSIs
    Output: Absolute path for each folder /path/WSI1, /path/WSI2, etc.
    """
    input_folders = config.input_folders
    input_folders_list = []
    
    # To process all the images in the input and sub folders
    if input_folders is None and config.sub_folders is None and config.process_all:  
        input_folders_list.append(input_path)
        return input_folders_list 
    
    elif input_folders is None: 
        # Process all input folders in the input path 
        for item in os.listdir(input_path):
            item_path = os.path.join(input_path, item)
            if os.path.isdir(item_path): 
                input_folders_list.append(item_path)

        # If no subdirectories found, check for image files in the current folder
        if not input_folders_list:
            has_images = any(
                os.path.isfile(os.path.join(input_path, f)) and f.lower().endswith(IMAGE_EXTENSIONS)
                for f in os.listdir(input_path)
            )
            if has_images:
                print(f"No subfolders found, but images exist in '{input_path}'. Using it as target.")
                return [input_path]
            else:
                return []  # No subfolders and no images found 
        return input_folders_list 
    
    else:
        # Process only specified folders
        input_folder_names = [name.strip() for name in config.input_folders.split(',')] 
        for name in input_folder_names:
            folder_path = os.path.join(input_path, name)
            if os.path.isdir(folder_path):
                input_folders_list.append(folder_path)
            else:
                print(f"Warning: Folder '{folder_path}' does not exist. Skipping.")
        return input_folders_list
       
def process_all_input_folders(config: Config, device: torch.device, model_path: str):
    try:
        # Load model with error handling
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return
            
        model = Auto_encoder.AutoEncoder().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        input_folders_list = get_input_folders_list(config.input_path, config)
        #print("input_folders_list : ",input_folders_list)

        if not input_folders_list:
            print(f"No folders found to process in {config.input_path}")
            return 
        
        print(f"Found {len(input_folders_list)} input folders (WSIs) to process: "f"{', '.join(os.path.basename(s) for s in input_folders_list)}")
        print("-----------------------------------------------------")
        # Process each subfolder
        successful = 0
        failed = 0 
        
        for input_folder_path in input_folders_list:
            success = process_input_folder(input_folder_path, config, device, model)
            if success:
                successful += 1
            else:
                failed += 1
            
            # Clear memory after each subfolder
            if device.type == 'cuda':
                torch.cuda.empty_cache() 
                
        # Clean up any remaining temporary directories
        temp_features_base = os.path.join(config.output_path, 'temp_features')
        if os.path.exists(temp_features_base):
            try:
                import shutil
                shutil.rmtree(temp_features_base)
                print(f"\nCleaned up all temporary feature directories")
            except Exception as e:
                print(f"Warning: Could not clean up temporary feature base directory: {e}")
        
        print("\nAll processing completed!")
        print(f"\nSuccessfully processed: {successful} input_folders (WSIs)\n\n")
        if failed > 0:
            print(f"Failed to process: {failed} input_folders (WSIs)")
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        import traceback
        traceback.print_exc()
          
# ====================================
# Main
# ====================================
def str2bool(s: str) -> bool:
    if s.lower() in {'yes', 'true', 't', '1'}:
        return True
    if s.lower() in {'no', 'false', 'f', '0'}:
        return False
    raise argparse.ArgumentTypeError('Expected a boolean value.')

def main():
    # Set multiprocessing start method to 'spawn' for better compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    parser = argparse.ArgumentParser(description='Cluster and sample WSI images from subfolders')
    
    # Required arguments
    parser.add_argument('--input_path', required=True, help='Path to the input folders (WSIs) containing subfolders with images')
    parser.add_argument('--input_folders', type=str, default=None, help='Comma-separated input folder (WSI) names to process. If None, all folders (WSI) will be processed.')
    parser.add_argument('--sub_folders', type=str, default=None, help='Comma-separated sub-folder names in the input folders. If None, all sub_folders will be processed.')
    parser.add_argument('--process_all', type=str2bool, default=False, help='True will switch to process all images in all the input folders. False by default turns it offf.')   
    parser.add_argument('--output_path', required=True, help='Path to the output folder')   

    # Optional arguments
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for feature extraction (default: 64)') 
    parser.add_argument('--dim_reduce', type=int, default=256, help='Dimensionality reduction before clustering (default: 256)') 
    parser.add_argument('--distance_groups', type=int, default=5, help='Number of distance groups for sampling (default: 5)')
    parser.add_argument('--sample_percentage', type=float, default=0.20,  help='Percentage of images to sample from each group (default: 0.20)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Device to use (auto, cpu, cuda) (default: auto)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use if CUDA is selected (default: 0)')
    parser.add_argument('--model', type=str,  default="AE_CRC.pth", help='Path to the model file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)') 

    # Enabling/disabling output folders
    parser.add_argument('--store_features', type=str2bool, default=False, help='Store features and cluster assignment files permanently (default: True)')
    parser.add_argument('--store_clusters', type=str2bool, default=False, help='Create clusters folder with clustered images (default: True)')
    parser.add_argument('--store_plots', type=str2bool, default=False, help='Create plots folder with visualization plots (default: True)')
    parser.add_argument('--store_samples', type=str2bool, default=False, help='Create samples folder with sampled images (default: True)')
    parser.add_argument('--store_samples_group_wise', type=str2bool, default=False, help='Store samples in group-wise folders within clusters (default: True)')
    
    args = parser.parse_args() 
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed) 
        
    # Create config object
    config = Config(
        input_path = args.input_path,
        input_folders = args.input_folders, 
        sub_folders = args.sub_folders, 
        output_path = args.output_path, 
        process_all = args.process_all, 
        batch_size = args.batch_size,
        dim_reduce = args.dim_reduce,
        num_distance_groups= args.distance_groups,
        sample_percentage = args.sample_percentage,
        store_features = args.store_features,
        store_clusters = args.store_clusters,
        store_plots = args.store_plots,
        store_samples = args.store_samples,
    	store_samples_group_wise = args.store_samples_group_wise
    )  

    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'cpu':
        device = torch.device('cpu')
    else:  # args.device == 'cuda'
        if not torch.cuda.is_available():
            print("CUDA requested but not available. Using CPU instead.")
            device = torch.device('cpu')
        else:
            if args.gpu_id >= torch.cuda.device_count():
                print(f"GPU {args.gpu_id} requested but not available. Using GPU 0 instead.")
                device = torch.device('cuda:0')
            else:
                device = torch.device(f'cuda:{args.gpu_id}')
    
    print("-----------------------------------------------------")
    if torch.cuda.is_available() and 'cuda' in str(device):
        print(f"Using device: {device} {torch.cuda.get_device_name(device)}")
    print("-----------------------------------------------------")
    # Process the subfolders
    process_all_input_folders(config, device, args.model)

if __name__ == "__main__":
    main()