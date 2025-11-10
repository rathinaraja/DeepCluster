# Standard library imports
import os
import time
import math 
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Machine learning imports for CPU version
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull 

# GPU libraries (optional)
GPU_AVAILABLE = False
try:
    import cupy as cp
    from cuml.cluster import KMeans as cuKMeans
    from cuml.decomposition import PCA as cuPCA
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Custom imports
from Config import Config

# Calculating the number of clusters
def calc_num_clusters(n_samples):
    a = 1   
    n_clusters = math.ceil(a * math.sqrt(n_samples))
    return n_clusters

# Performing k-means
def perform_kmeans(X, n_clusters, folder_name, use_gpu, device, actual_gpu_id=None):
    
    device_display = f"GPU:{actual_gpu_id}" if actual_gpu_id is not None else str(device)    
    progress_desc = f"|{folder_name}| - |{device_display}| - |K-means ({n_clusters} clusters)|"    
    
    # Decide whether to use GPU
    can_use_gpu = use_gpu and GPU_AVAILABLE and device.type == 'cuda'
    
    if can_use_gpu:
        # GPU path
        with tqdm(total = 100, desc = f"{progress_desc}", unit = "%", leave = True, ncols = 120,
                  bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            try:
                # Transfer to GPU
                X_gpu = cp.asarray(X, dtype=cp.float32)
                pbar.n = 20
                pbar.refresh()  
                
                # GPU K-means
                kmeans = cuKMeans(n_clusters=n_clusters, random_state=42, max_iter=300)
                clusters_gpu = kmeans.fit_predict(X_gpu)
                
                pbar.n = 80
                pbar.refresh()     
                
                # Transfer back to CPU
                clusters = cp.asnumpy(clusters_gpu)                
                pbar.n = 100
                pbar.refresh()
                
                return clusters
                
            except Exception as e:
                print(f"  GPU clustering failed: {e}, falling back to CPU")
                can_use_gpu = False
    
    # CPU path (fallback or default)
    if not can_use_gpu:
        with tqdm(total = 100, desc = f"{progress_desc}", unit = "%", leave = True, ncols = 120,
                  bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            
            pbar.n = 20
            pbar.refresh()            
            kmeans = MiniBatchKMeans(n_clusters = n_clusters, random_state = 42, batch_size = min(2048, len(X)), max_iter = 100, n_init = 3)      
            
            pbar.n = 50
            pbar.refresh()
            
            clusters = kmeans.fit_predict(X)            
            pbar.n = 100
            pbar.refresh()
            
            return clusters

def perform_pca(features_scaled, n_components, folder_name, use_gpu, device, actual_gpu_id=None): 
    
    device_display = f"GPU:{actual_gpu_id}" if actual_gpu_id is not None else str(device)    
    progress_desc = f"|{folder_name}| - |{device_display}| - |PCA (512 -> {n_components} dims)|"    
    
    # Decide whether to use GPU
    can_use_gpu = use_gpu and GPU_AVAILABLE and device.type == 'cuda'
    
    if can_use_gpu:
        # GPU path
        with tqdm(total = 1, desc = f"{progress_desc}", unit="step", leave = True, ncols = 120,
                  bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            try:
                # Transfer to GPU
                features_gpu = cp.asarray(features_scaled, dtype=cp.float32)                
                # GPU PCA
                pca = cuPCA(n_components=n_components, random_state=42)
                features_reduced_gpu = pca.fit_transform(features_gpu)                
                # Transfer back to CPU
                features_reduced = cp.asnumpy(features_reduced_gpu)                
                pbar.update(1)                
                return features_reduced
                
            except Exception as e:
                print(f"  GPU PCA failed: {e}, falling back to CPU")
                can_use_gpu = False
    
    # CPU path (fallback or default)
    if not can_use_gpu:
        with tqdm(total = 1, desc = f"{progress_desc}", unit = "step", leave = True, ncols = 120,
                  bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            pca = PCA(n_components=n_components, random_state=42)
            features_reduced = pca.fit_transform(features_scaled)
            pbar.update(1)
            
            return features_reduced

# Perform PCA for 2D visualization
def perform_pca_visualization(features_reduced_pca, folder_name, use_gpu, device, actual_gpu_id=None): 
    
    n_components = features_reduced_pca.shape[1]

    device_display = f"GPU:{actual_gpu_id}" if actual_gpu_id is not None else str(device)
    progress_desc = f"|{folder_name}| - |{device_display}| - |PCA Visualization ({n_components} -> 2 dims)|"    
    
    # Decide whether to use GPU
    can_use_gpu = use_gpu and GPU_AVAILABLE and device.type == 'cuda'
    
    if can_use_gpu:
        # GPU PCA path
        with tqdm(total=1, desc=f"{progress_desc}", unit="step", leave=True, ncols=120,
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            try:
                # Transfer to GPU
                features_gpu = cp.asarray(features_reduced_pca, dtype=cp.float32)                
                # GPU PCA for 2D visualization
                pca = cuPCA(n_components=2, random_state=42)
                features_2d_gpu = pca.fit_transform(features_gpu)                
                # Transfer back to CPU
                features_2d = cp.asnumpy(features_2d_gpu)                
                pbar.update(1)                
                return features_2d
                
            except Exception as e:
                print(f"  GPU PCA visualization failed: {e}, falling back to CPU")
                can_use_gpu = False
    
    # CPU PCA path (fallback or default)
    if not can_use_gpu:
        with tqdm(total=1, desc=f"{progress_desc}", unit="step", leave=True, ncols=120,
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            
            # CPU PCA for 2D visualization
            pca = PCA(n_components=2, random_state=42)
            features_2d = pca.fit_transform(features_reduced_pca)
            
            pbar.update(1)
            return features_2d

def copy_image_parallel(args):
    """Helper function to copy a single image"""
    file_path, output_path = args
    try:
        if os.path.exists(file_path):
            img = Image.open(file_path)
            img.save(output_path)
            return True
            
    except Exception as e:
        print(f"Error processing image {file_path}: {e}")
        
    return False

# Invoking cluster processing
def process_clusters(folder_name: str, feature_file: str, n_samples: int, config: Config, temp_feature_dir: str, device, actual_gpu_id=None):
    
    device_display = f"GPU:{actual_gpu_id}" if actual_gpu_id is not None else str(device)
    
    try: 
        # Load data
        data = pd.read_csv(feature_file)
        features = data.drop('File_Path', axis=1).values.astype(np.float32)
        file_paths = data['File_Path'].values 
        
        # Scaling
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        features_scaled = np.nan_to_num(features_scaled, nan=0.0001, posinf=0.0001, neginf=0.0001)        
             
        # PCA for dimensionality reduction
        n_components = config.dim_reduce   
        if n_samples <= n_components:
            n_clusters = 1
            clusters = np.zeros(n_samples, dtype=int) 
            features_reduced_pca = features_scaled
        else:
            # Perform PCA (with optional GPU)
            features_reduced_pca = perform_pca(features_scaled, n_components, folder_name, config.use_gpu_clustering, device, actual_gpu_id)            
            n_clusters = calc_num_clusters(n_samples) 
            
        # K-means clustering (with optional GPU)
        if n_samples > n_components:
            clusters = perform_kmeans(features_reduced_pca, n_clusters, folder_name, config.use_gpu_clustering, device, actual_gpu_id)

        # Visualization (if enabled) - NOW USING PCA INSTEAD OF t-SNE
        if config.store_plots and n_samples > 2 and n_clusters > 1:         
            plot_dir = os.path.join(config.plot_dir, folder_name)
            os.makedirs(plot_dir, exist_ok=True)
    
            # Clean data for visualization
            features_reduced_pca = np.nan_to_num(features_reduced_pca, nan=0.0001, posinf=0.0001, neginf=0.0001)
            
            # Perform PCA for 2D visualization (replaces t-SNE) 
            features_2d = perform_pca_visualization(features_reduced_pca, folder_name, config.use_gpu_clustering, device, actual_gpu_id)
            
            palette = sns.color_palette("husl", n_colors=n_clusters)
            visualize_clusters(features_2d, clusters, file_paths, os.path.join(plot_dir, "pca_visualization"), palette)  
        
        assignments = pd.DataFrame({'File_Path': file_paths, 'Cluster': clusters})
            
        # Save cluster assignments
        if config.store_features:
            feature_dir = os.path.dirname(feature_file)
            assignments.to_csv(os.path.join(feature_dir, "cluster_assignments.csv"), index=False)
        else:
            assignments.to_csv(os.path.join(temp_feature_dir, "cluster_assignments.csv"), index=False)
            
        if config.store_clusters:
            cluster_base = os.path.join(config.cluster_dir, folder_name)
            os.makedirs(cluster_base, exist_ok=True)
    
            progress_desc = f"|{folder_name}| - |{device_display}| - |Saving cluster data|"
            
            # Parallel image copying
            max_workers = min(8, mp.cpu_count())
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for cluster in tqdm(range(n_clusters), desc = progress_desc, unit = "cluster", leave = True, ncols = 120,
                                   bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
                    
                    cluster_dir = os.path.join(cluster_base, f'Cluster_{cluster}')
                    os.makedirs(cluster_dir, exist_ok=True)                    
                    cluster_files = file_paths[clusters == cluster]
                    
                    copy_args = [
                        (file_path, os.path.join(cluster_dir, os.path.basename(file_path)))
                        for file_path in cluster_files if os.path.exists(file_path)
                    ]                    
                    list(executor.map(copy_image_parallel, copy_args))
        
        return assignments, n_clusters
        
    except Exception as e:
        print(f"Error in process_clusters: {str(e)}")
        raise

def visualize_clusters(features_2d, clusters, file_paths, output_prefix: str, palette):
    """Visualization function - now using PCA instead of t-SNE"""
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    plt.switch_backend('Agg')
    
    # Plot with legend
    plt.figure(figsize=(12, 8))
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], c=[palette[cluster]], 
                   label=f'Cluster {cluster}', alpha=0.6)
        
        if np.sum(mask) >= 5:
            try:
                points = features_2d[mask]
                x_range = np.ptp(points[:, 0])
                y_range = np.ptp(points[:, 1])
                
                if x_range > 1e-6 and y_range > 1e-6:
                    hull = ConvexHull(points)
                    for simplex in hull.simplices:
                        plt.plot(points[simplex, 0], points[simplex, 1], c=palette[cluster])
            except:
                pass
    
    plt.title("PCA Visualization with K-Means Clusters")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_with_legend.png", dpi=400, bbox_inches='tight') 
    plt.close()
    plt.clf()
    
    # Plot with numbers
    plt.figure(figsize=(16, 11))
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], c=[palette[cluster]], alpha=0.6)
        
        if np.sum(mask) > 0:
            center = features_2d[mask].mean(axis=0)
            plt.text(center[0], center[1], str(cluster), 
                    horizontalalignment='center', verticalalignment='center', fontweight='bold')
        
        if np.sum(mask) >= 5:
            try:
                points = features_2d[mask]
                x_range = np.ptp(points[:, 0])
                y_range = np.ptp(points[:, 1])
                
                if x_range > 1e-6 and y_range > 1e-6:
                    hull = ConvexHull(points)
                    for simplex in hull.simplices:
                        plt.plot(points[simplex, 0], points[simplex, 1], c=palette[cluster])
            except:
                pass
    
    plt.title("PCA Visualization with Cluster Numbers")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_with_numbers.png", dpi=400, bbox_inches='tight') 
    plt.close()
    plt.clf()