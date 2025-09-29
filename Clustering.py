# Standard library imports
import os
import time
import threading
import math 
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning imports
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE 
from scipy.spatial import ConvexHull 

# Custom imports (these need to be available in your project)
from Config import Config  # Your configuration class

# Calculating the number of samples using square root
def calc_num_clusters(n_samples):
    # Number of clusters adjustment (less cluster more samples per cluster) 0 - 1 (more cluster less samples per cluster)
    a = 1   
    n_clusters = math.ceil(a * math.sqrt(n_samples))
    return n_clusters

# Choosing perplexity for t-SNE
def choose_perplexity(n_samples):
    if n_samples < 50:
        return min(15, n_samples - 1)
    elif n_samples < 500:
        return 30
    elif n_samples < 10000:
        return 50
    else:
        return 100  # Can go higher for very large datasets

# Progress bar for kmeans
class ProgressKMeans:
    """Wrapper class for KMeans with progress tracking"""
    
    def __init__(self, folder_name, n_clusters, random_state=42, n_init=10):
        self.folder_name = folder_name
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init 
        self.progress_bar = None
        self.stop_progress = False
        
    def _progress_updater(self, description):
        """Background thread to update progress bar"""
        self.progress_bar = tqdm( total = 100, desc = description, unit = "%", leave = True, ncols = 120, bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')        
        # Simulate progress updates (since we can't track actual K-means iterations)
        progress_points = [10, 25, 40, 60, 80, 95]
        for progress in progress_points:
            if self.stop_progress:
                break
            time.sleep(0.5)  # Small delay between updates
            self.progress_bar.n = progress
            self.progress_bar.refresh()
        
    def fit_predict(self, X):
        """Fit K-means and predict clusters with progress tracking"""
        # Start progress tracking in background thread
        progress_desc = f"|{self.folder_name}| - K-means ({self.n_clusters} clusters)"
        progress_thread = threading.Thread( target = self._progress_updater,  args = (progress_desc,))
        progress_thread.daemon = True
        progress_thread.start()
        
        try:
            # Perform actual K-means clustering
            kmeans = KMeans( n_clusters = self.n_clusters, random_state = self.random_state, n_init = self.n_init)
            clusters = kmeans.fit_predict(X)
            
            # Stop progress updates and complete the bar
            self.stop_progress = True
            if self.progress_bar:
                self.progress_bar.n = 100
                self.progress_bar.refresh()
                self.progress_bar.close()
            
            return clusters
            
        except Exception as e:
            # Ensure progress bar is closed on error
            self.stop_progress = True
            if self.progress_bar:
                self.progress_bar.close()
            raise e

#Performing clustering using kmeans
def process_clusters(folder_name: str, feature_file: str, n_samples: int, config: Config, temp_feature_dir: str): 
    try: 
        data = pd.read_csv(feature_file)         
        features = data.drop('File_Path', axis=1).values
        file_paths = data['File_Path'].values 
        
        #print("Scaling features...")
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Cleaning NaN and infinity values 
        features_scaled = np.nan_to_num(features_scaled, nan=0.0001, posinf=0.0001, neginf=0.0001)        
             
        # PCA with progress indication
        n_components = config.dim_reduce   
        if n_samples <= n_components:
            # Use single cluster when samples <= components to avoid Intel oneMKL error
            n_clusters = 1
            clusters = np.zeros(n_samples, dtype=int) 
            features_reduced_pca = features_scaled  # Skip PCA, use scaled features
        else:
            #print(f"Applying PCA (512 -> {n_components} dimensions)...")
            progress_desc = f"|{folder_name}| - PCA (512 -> {n_components} dimensions)"
            with tqdm(total = 1, desc = progress_desc, unit = "step", leave = True, ncols = 120, bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                pca = PCA(n_components = n_components) 
                features_reduced_pca = pca.fit_transform(features_scaled)
                pbar.update(1)                
            n_clusters = calc_num_clusters(n_samples) 
            
        # K-means clustering with progress bar
        #print(f"Performing K-means clustering ({n_clusters} clusters)...")
        progress_kmeans = ProgressKMeans(folder_name = folder_name, n_clusters = n_clusters, random_state = 42, n_init = 10)
        clusters = progress_kmeans.fit_predict(features_reduced_pca)

        if config.store_plots and n_samples > 2 and n_clusters > 1:         
            # Create subfolder-specific plot directory
            plot_dir = os.path.join(config.plot_dir, folder_name)
            os.makedirs(plot_dir, exist_ok=True)

            # Calculate safe perplexity value (must be less than n_samples)
            #perplexity = min(10, n_samples // 3) 
    
            progress_desc = f"|{folder_name}| - t-SNE ({n_components} -> 2 dimensions)"
            features_reduced_pca = np.nan_to_num(features_reduced_pca, nan=0.0001, posinf=0.0001, neginf=0.0001)    
            with tqdm(total = 1, desc = progress_desc, unit = "step", leave = True, ncols = 120, bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]' ) as pbar:
                tsne = TSNE(n_components = 2, random_state = 42, perplexity = choose_perplexity(n_samples), n_iter = 1500, learning_rate = 'auto', n_jobs = 1)
                features_2d_tsne = tsne.fit_transform(features_reduced_pca)
                pbar.update(1)
            palette = sns.color_palette("husl", n_colors=n_clusters)
            visualize_clusters(features_2d_tsne, clusters, file_paths, os.path.join(plot_dir, "tsne"), palette)  
        
        assignments = pd.DataFrame({ 'File_Path': file_paths, 'Cluster': clusters })
            
        # Save cluster assignments based on store_features flag
        if config.store_features:
            feature_dir = os.path.dirname(feature_file)
            assignments.to_csv(os.path.join(feature_dir, "cluster_assignments.csv"), index=False)
        else:
            # Save to temporary directory for sampling process
            assignments.to_csv(os.path.join(temp_feature_dir, "cluster_assignments.csv"), index=False)
            
        if config.store_clusters:
            # Create subfolder-specific cluster directory
            cluster_base = os.path.join(config.cluster_dir, folder_name)
            os.makedirs(cluster_base, exist_ok=True)
    
            #print("Organizing images into cluster folders...")
            progress_desc = f"|{folder_name}| - Saving cluster data"
            for cluster in tqdm(range(n_clusters), desc = progress_desc, unit = "cluster", leave = True, ncols = 120, bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
                cluster_dir = os.path.join(cluster_base, f'Cluster_{cluster}')
                os.makedirs(cluster_dir, exist_ok=True)
                
                cluster_files = file_paths[clusters == cluster]  
                
                for file_path in cluster_files:
                    try:  
                        if os.path.exists(file_path):
                            img = Image.open(file_path)
                            img.save(os.path.join(cluster_dir, os.path.basename(file_path)))
                        else:
                            print(f"File not found: {file_path}")
                    except Exception as e:
                        print(f"Error processing image {file_path}: {e}")
            
            # Print the actual samples per cluster distribution 
            cluster_counts = np.bincount(clusters)
            avg_samples_per_cluster = n_samples / n_clusters if n_clusters > 0 else 0
            #print(f"Clustering complete - Avg {avg_samples_per_cluster:.1f} samples/cluster")
            #print(f"Distribution: Min: {cluster_counts.min()}, Max: {cluster_counts.max()} samples")
        
        #print(f"Clustering completed for {folder_name}\n")
        return assignments, n_clusters
        
    except Exception as e:
        print(f"Error in process_clusters: {str(e)}")
        raise

def visualize_clusters(features_2d, clusters, file_paths, output_prefix: str, palette):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    # Use matplotlib's Agg backend for headless environments
    plt.switch_backend('Agg')
    
    # Plot with legend
    plt.figure(figsize=(12, 8))
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], c = [palette[cluster]], label = f'Cluster {cluster}', alpha = 0.6 )
        
        # More conservative approach - only draw hulls for larger, well-separated clusters
        if np.sum(mask) >= 5:  # Require more points
            try:
                points = features_2d[mask]
                # Check if points have sufficient spread
                x_range = np.ptp(points[:, 0])  # peak-to-peak range
                y_range = np.ptp(points[:, 1])
                
                if x_range > 1e-6 and y_range > 1e-6:  # Minimum spread threshold
                    hull = ConvexHull(points)
                    for simplex in hull.simplices:
                        plt.plot(points[simplex, 0], points[simplex, 1], c=palette[cluster])
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
    
    # Plot with cluster numbers - APPLY THE SAME FIX HERE
    plt.figure(figsize=(16, 11))
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        plt.scatter( features_2d[mask, 0], features_2d[mask, 1], c = [palette[cluster]], alpha = 0.6 )
        
        # Add cluster number in center
        if np.sum(mask) > 0:  # Only if cluster has points
            center = features_2d[mask].mean(axis=0)
            plt.text(center[0], center[1], str(cluster),  horizontalalignment = 'center', verticalalignment = 'center', fontweight = 'bold')
        
        # FIXED: Apply the same conservative approach as the first plot
        if np.sum(mask) >= 5:  # Require more points
            try:
                points = features_2d[mask]
                # Check if points have sufficient spread
                x_range = np.ptp(points[:, 0])  # peak-to-peak range
                y_range = np.ptp(points[:, 1])
                
                if x_range > 1e-6 and y_range > 1e-6:  # Minimum spread threshold
                    hull = ConvexHull(points)
                    for simplex in hull.simplices:
                        plt.plot(points[simplex, 0], points[simplex, 1], c=palette[cluster])
            except Exception as e:
                #print(f"Could not create convex hull for cluster {cluster}: {e}")
                pass
    
    plt.title("t-SNE Visualization with Cluster Numbers")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_with_numbers.png", dpi=400, bbox_inches='tight') 
    plt.close()
    plt.clf()  # Clear figure memory