import os, time, math
import glob, csv  
import shutil
from tqdm import tqdm  
from datetime import datetime
from pathlib import Path
import traceback

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
from PIL import Image 

import torch
import torch.nn as nn 
from torch.utils.data import Dataset 
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.multiprocessing as mp 

import Auto_encoder 
from Dataset import WSIDataset, count_images_input_folder
from Feature_extraction import extract_features
from Clustering import calc_num_clusters, process_clusters
from Sampling import sample_clustered_images
from Logging import write_metrics_to_log, write_metrics_to_log_serial 

image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}

def get_actual_gpu_id(pytorch_gpu_id):
    """Convert PyTorch GPU ID to actual hardware GPU ID"""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible:
        visible_gpus = [int(x.strip()) for x in cuda_visible.split(',') if x.strip().isdigit()]
        if pytorch_gpu_id < len(visible_gpus):
            return visible_gpus[pytorch_gpu_id]
    return pytorch_gpu_id  # Fallback to PyTorch ID

# checking subfolders
def has_no_subfolders(path) -> bool:
    p = Path(path)
    return p.is_dir() and not any(child.is_dir() for child in p.iterdir())

# Loading AutoEncoder model
def load_ae_model(config, device):
    """Load autoencoder model"""
    model = Auto_encoder.AutoEncoder().to(device)
    model.load_state_dict(torch.load("AE_CRC.pth", map_location=device))
    model.eval()
    return model 

# Counting the number of images
def count_images(input_path):    
    counter = 0
    if os.path.exists(input_path):
    # Count all image files in the samples folder (including subfolders if store_samples_group_wise is True)    
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    counter += 1
    return counter

# DeepCluster algorithm
def DeepCluster(slide_path, device, config, log_file, ae_model, actual_gpu_id=None, csv_lock=None):
    # Extract features
    folder_name = os.path.basename(slide_path)
    temp_feature_dir = None 
    # Initialize timing and metrics
    start_time = time.time()
    feature_extraction_time = 0.0
    clustering_time = 0.0
    sampling_time = 0.0

    try:  
        # First count the number of images to estimate the number of clusters
        n_samples = count_images_input_folder(slide_path, config)

        if n_samples == 0:
            print(f"No images found in folder: {slide_path}. Skipping.")
            return False
  
        # Pre-calculate and display the expected number of clusters 
        n_clusters = calc_num_clusters(n_samples)
        per_cluster = n_samples / n_clusters  
        
        # Display with actual GPU ID
        device_display = f"GPU:{actual_gpu_id}" if actual_gpu_id is not None else str(device)
        print(f"\n|{folder_name}| - |{device_display}| - |{n_samples} samples| - |Processing is started!!!|")

        feature_start = time.time()
        result = extract_features(slide_path, folder_name, n_samples, config, device, ae_model, actual_gpu_id)
        feature_extraction_time = time.time() - feature_start
        
        if len(result) == 3:
            feature_file, actual_samples, temp_feature_dir = result
        else:
            feature_file, actual_samples = result
            temp_feature_dir = None
      
        if feature_file is not None:
            # Process clusters
            clustering_start = time.time()
            assignments, n_clusters = process_clusters(folder_name, feature_file, actual_samples, config, temp_feature_dir or "")
            clustering_time = time.time() - clustering_start
            
            # Sample representative images
            sampling_start = time.time()
            num_samples_selected = 0
            if config.store_samples: 
                sample_clustered_images(folder_name, n_clusters, config, temp_feature_dir or "")
                # Count the actual number of sampled images
                samples_folder = os.path.join(config.output_path, "samples", folder_name)
                num_samples_selected = count_images(samples_folder) 
            sampling_time = time.time() - sampling_start

            # Calculate overall_time and now_local
            overall_time = time.time() - start_time
            now_local = datetime.now().astimezone() 
       
            # Prepare metrics dictionary - USE actual_samples instead of n_samples
            metrics_dict = {
                'total_samples': actual_samples,  # Changed from n_samples to actual_samples
                'num_clusters': n_clusters,
                'num_samples_selected': num_samples_selected,
                'feature_extraction_time': round(feature_extraction_time, 2),
                'clustering_time': round(clustering_time, 2),
                'sampling_time': round(sampling_time, 2),
                'overall_time': round(overall_time, 2),
                'date_time_processed': str(now_local.strftime("%Y-%m-%d %H:%M:%S %Z")),
                'additional_note': f'GPU:{actual_gpu_id}' if actual_gpu_id is not None else 'CPU'
            }
            
            # Add sub_folder counts metrics if applicable
            if config.sub_folders:
                sub_folder_names = [name.strip() for name in config.sub_folders.split(',')]
                sub_folder_counts = []
                for sub_folder in sub_folder_names:
                    sub_path = os.path.join(slide_path, sub_folder)
                    if os.path.exists(sub_path):
                        # Count images in this specific sub_folder using the existing count_images_in_folder function
                        sub_count = count_images(sub_path)
                        sub_folder_counts.append(sub_count)
                    else:
                        sub_folder_counts.append(0)
                metrics_dict['sub_folder_counts'] = sub_folder_counts 
            
            # Write metrics to log file while running on cpu
            if str(device) == 'cpu': 
                write_metrics_to_log_serial(log_file, folder_name, metrics_dict)

            # Write metrics to log file while running on gpu with multiprocessing lock
            if csv_lock: 
                write_metrics_to_log(log_file, folder_name, metrics_dict, csv_lock)
                
            print(f"\n|{folder_name}| - |{device_display}| - |{n_samples} samples| - |Processing is completed!!!|")  
            
            # Clean up temporary files if needed
            if temp_feature_dir:  
                cleanup_temporary_files(temp_feature_dir, config)            
            return True
        else:
            print(f"Feature extraction failed for subfolder: {folder_name}. Skipping.")
            return False
            
    except Exception as e:
        print(f"Error processing WSI {folder_name}: {str(e)}")         
        # Clean up temporary files in case of error
        if temp_feature_dir:
            cleanup_temporary_files(temp_feature_dir, config)        
        return False
        
    finally:
        # Clear GPU memory for this worker
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        if temp_feature_dir: 
            cleanup_temporary_files(temp_feature_dir, config)      

# Processing input folders
def process_input_folder_on_gpu(slides, gpu_id, device, config, log_file, ae_model, gpu_mapping=None, csv_lock=None):
    """Process slides on a specific GPU with actual GPU ID display"""
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(gpu_id) 
    
    # Get actual hardware GPU ID for display
    actual_gpu_id = gpu_mapping.get(gpu_id, gpu_id) if gpu_mapping else get_actual_gpu_id(gpu_id)
    
    results = []

    for slide_path in slides:
        try:
            result = DeepCluster(slide_path, device, config, log_file, ae_model, actual_gpu_id, csv_lock)
            results.append(result)
        except Exception as e:
            print(f"\nError processing {slide_path} on GPU {actual_gpu_id}: {e}")
    return results

# Displaying the original GPU ID
def create_gpu_mapping(specified_gpu_ids):
    """Create mapping from PyTorch GPU indices to actual hardware GPU IDs"""
    if specified_gpu_ids is None:
        # If no specific GPUs specified, use all available
        return {i: i for i in range(torch.cuda.device_count())}
    
    # Create mapping from PyTorch index to actual GPU ID
    gpu_mapping = {}
    for pytorch_idx, actual_gpu_id in enumerate(specified_gpu_ids):
        gpu_mapping[pytorch_idx] = actual_gpu_id
    
    return gpu_mapping

# Assigning list of input folders to multiple GPUs
def process_all_input_folders_parallel(config, device, log_file, specified_gpu_ids=None):
    """Process all slides using multiple GPUs with proper GPU ID mapping"""
    try:
        # IMPORTANT: Set the multiprocessing start method FIRST, before creating any locks
        mp.set_start_method('spawn', force=True)        
        # Create the CSV lock AFTER setting spawn method
        csv_lock = mp.Lock()
        
        # Prepare dataset
        if has_no_subfolders(config.input_path) or config.process_all:
            input_folders_list = [config.input_path]          
        else:
            dataset = WSIDataset(config.input_path, config.selected_input_folders, log_file)
            input_folders_list = dataset.slide_files 
    
        if not input_folders_list:
            print(f"No folders found to process in {config.input_path}")
            return
    
        # Create GPU mapping
        gpu_mapping = create_gpu_mapping(specified_gpu_ids)        
        # Get available GPUs (PyTorch indices)
        if specified_gpu_ids:
            # Set CUDA_VISIBLE_DEVICES to the specified GPUs
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, specified_gpu_ids))
            device_ids = list(range(len(specified_gpu_ids)))
        else:
            device_ids = list(range(torch.cuda.device_count()))

        num_gpus = min(len(device_ids), len(input_folders_list))     
        # Distribute slides across GPUs
        slides_split = [[] for _ in range(num_gpus)]
        
        # Assign one slide to each GPU first
        for i in range(num_gpus):
            slides_split[i].append(input_folders_list[i])
     
        # Distribute remaining slides
        remaining_slides = input_folders_list[num_gpus:]
        for idx, slide in enumerate(remaining_slides):
            slides_split[idx % num_gpus].append(slide)
        
        # Show slide distribution with actual GPU IDs
        print("=== WSIs (input folders) Distribution ===")
        for pytorch_gpu_id in range(num_gpus):
            actual_gpu_id = gpu_mapping.get(pytorch_gpu_id, pytorch_gpu_id)
            
            slide_names = [os.path.basename(slide).split(',')[0].strip() for slide in slides_split[pytorch_gpu_id]] 
            print(f"\nHardware GPU {actual_gpu_id}: {len(slides_split[pytorch_gpu_id])} slides - {slide_names}") 
        print('-' * 100)
        print("DeepCluster++ started...!!!")  
        print('-' * 100 )

        # Load autoencoder model
        ae_model = load_ae_model(config, device)
    
        # Launch processing on each GPU
        processes = []
        
        for pytorch_gpu_id in range(num_gpus):
            process = torch.multiprocessing.Process(
                target=process_input_folder_on_gpu,
                args=(slides_split[pytorch_gpu_id], pytorch_gpu_id, device, config, log_file, ae_model, gpu_mapping, csv_lock)
            )
            processes.append(process)
            process.start()
    
        # Wait for completion
        for process in processes:
            process.join() 

        print("\nDeepCluster++ completed...!!!")  
        print('-' * 100 )
        # Clean up any remaining temporary directories
        temp_features_base = os.path.join(config.output_path, 'temp_features')
        if os.path.exists(temp_features_base):
            try: 
                shutil.rmtree(temp_features_base)
                print(f"\nCleaned up all temporary feature directories")
            except Exception as e:
                print(f"Warning: Could not clean up temporary feature base directory: {e}")
                    
        print("\nAll processing completed!!!\n")

    except Exception as e:
        print(f"\nError during processing: {str(e)}")        
        traceback.print_exc()

# Processing input folders with CPU
def process_all_input_folders_serial(config, device, log_file):
    # Load autoencoder model
    try:         
        # Create the CSV lock AFTER setting spawn method 
        ae_model = load_ae_model(config, device)  
    
        # Prepare dataset
        if has_no_subfolders(config.input_path) or config.process_all:
            input_folders_list = [config.input_path]          
        else:
            dataset = WSIDataset(config.input_path, config.selected_input_folders, log_file)
            input_folders_list = dataset.slide_files 
    
        if not input_folders_list:
            print(f"No folders found to process in {config.input_path}")
            return
            
        successful = 0
        failed = 0 
 
        print("DeepCluster++ started...!!!")  
        print('-' * 100 )
            
        for input_folder_path in input_folders_list:
            success = DeepCluster(input_folder_path, device, config, log_file, ae_model, actual_gpu_id=None, csv_lock=None)
            if success:
                successful += 1
            else:
                failed += 1 
        print("\nDeepCluster++ completed...!!!")  
        print('-' * 100 )
        # Clean up any remaining temporary directories
        temp_features_base = os.path.join(config.output_path, 'temp_features')
        if os.path.exists(temp_features_base):
            try: 
                shutil.rmtree(temp_features_base)
                print(f"\nCleaned up all temporary feature directories")
            except Exception as e:
                print(f"Warning: Could not clean up temporary feature base directory: {e}")
            
        print("\nAll processing completed!!!")
        print(f"\nSuccessfully processed: {successful} input_folders (WSIs)\n")
        if failed > 0:
            print(f"Failed to process: {failed} input_folders (WSIs)\n")
            
    except Exception as e:
        print(f"\nError during processing: {str(e)}")        
        traceback.print_exc()

# Cleaning up all the temporary files
def cleanup_temporary_files(temp_feature_dir, config):
    """Clean up temporary feature files if store_features is False"""
    if os.path.exists(temp_feature_dir):
        try: 
            shutil.rmtree(temp_feature_dir)
            #print(f"Cleaned up temporary feature files from {temp_feature_dir}")
        except Exception as e:
            #print(f"Warning: Could not clean up temporary files from {temp_feature_dir}: {e}")
            pass