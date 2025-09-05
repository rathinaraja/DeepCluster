# Standard library imports 
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image

# Custom module imports
from Config import Config 

# Sampling images from the cluster
def sample_clustered_images(folder_name: str, n_clusters: int, config: Config, temp_feature_dir: str):     
    try:
        # Determine where to read files from based on store_features flag
        if config.store_features:
            subfolder_feature_dir = os.path.join(config.feature_dir, folder_name)
            assignment_file = os.path.join(subfolder_feature_dir, "cluster_assignments.csv")
            feature_file = os.path.join(subfolder_feature_dir, "features.csv")
        else:
            assignment_file = os.path.join(temp_feature_dir, "cluster_assignments.csv")
            feature_file = os.path.join(temp_feature_dir, "features.csv")
        
        if not os.path.exists(assignment_file) or not os.path.exists(feature_file):
            print(f"Missing required files for sample selection in {folder_name}")
            return
        
        # Load assignments and features
        assignments = pd.read_csv(assignment_file)
        features_df = pd.read_csv(feature_file)
        
        # Prepare output directory
        samples_base = os.path.join(config.output_path, 'samples', folder_name)
        os.makedirs(samples_base, exist_ok=True)
        
        # Process each cluster
        cluster_ids = assignments['Cluster'].unique()
 
        for cluster_id in tqdm(cluster_ids, desc=f"|{folder_name}| - Sampling images",unit = "cluster", leave = True, ncols = 120, bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
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