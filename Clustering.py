# ====================================
# Clustering.py
# ====================================
import os
import math
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from PIL import Image
from Config import Config

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