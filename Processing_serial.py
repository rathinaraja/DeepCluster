# ====================================
# Processing_parallel.py
# ====================================
import os
import concurrent.futures
from typing import List
import numpy as np
import torch
import Auto_encoder
from Dataset import count_images_in_folder
from Feature_extraction import extract_features
from Clustering import calc_num_clusters, process_clusters
from Sampling import sample_clustered_images 
from Config import Config

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

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
   
def cleanup_temporary_files(temp_feature_dir: str, config: Config):
    """Clean up temporary feature files if store_features is False"""
    if not config.store_features and os.path.exists(temp_feature_dir):
        try:
            import shutil
            shutil.rmtree(temp_feature_dir)
            print(f"Cleaned up temporary feature files from {temp_feature_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up temporary files from {temp_feature_dir}: {e}")