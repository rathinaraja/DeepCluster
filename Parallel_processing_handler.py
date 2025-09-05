# Standard library imports
import os
import shutil
import traceback
from pathlib import Path

# PyTorch imports
import torch
import torch.multiprocessing as mp

# Custom module imports  
import Auto_encoder   
from Dataset import WSIDataset   
from DeepCluster_framework import DeepCluster   
   
# Loading AutoEncoder model
def load_ae_model(config, device):
    """Load autoencoder model"""
    model = Auto_encoder.AutoEncoder().to(device)
    model.load_state_dict(torch.load("AE_CRC.pth", map_location=device))
    model.eval()
    return model 

# Displaying the original GPU ID
def create_gpu_mapping(specified_gpu_ids):
    """Create mapping from PyTorch GPU indices to actual hardware GPU IDs with validation"""
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return {}
    
    total_gpus = torch.cuda.device_count()
    print(f"Total GPUs available: {total_gpus}\n")
    
    if specified_gpu_ids is None:
        # If no specific GPUs specified, use all available
        return {i: i for i in range(total_gpus)}
    
    # VALIDATION: Check if specified GPU IDs are valid
    valid_gpu_ids = []
    invalid_gpu_ids = []
    
    for gpu_id in specified_gpu_ids:
        if 0 <= gpu_id < total_gpus:
            valid_gpu_ids.append(gpu_id)
        else:
            invalid_gpu_ids.append(gpu_id)
    
    if invalid_gpu_ids:
        print(f"WARNING: Invalid GPU IDs removed: {invalid_gpu_ids}")
        print(f"Available GPU range: 0-{total_gpus-1}")
    
    if not valid_gpu_ids:
        print("No valid GPU IDs specified, falling back to GPU 0")
        valid_gpu_ids = [0] if total_gpus > 0 else []
    
    # Create mapping from PyTorch index to actual GPU ID
    gpu_mapping = {}
    for pytorch_idx, actual_gpu_id in enumerate(valid_gpu_ids):
        gpu_mapping[pytorch_idx] = actual_gpu_id
    
    return gpu_mapping 
    
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
    
# Processing input folders
def process_input_folder_on_gpu(slides, gpu_id, device, config, log_file, ae_model, gpu_mapping=None, csv_lock=None):
    """Process slides on a specific GPU with validation and error handling"""
    
    try:
        # Validate GPU availability
        if not torch.cuda.is_available():
            print(f"CUDA not available in worker, using CPU")
            device = torch.device('cpu')
        elif gpu_id >= torch.cuda.device_count():
            print(f"Invalid GPU ID {gpu_id}, available GPUs: 0-{torch.cuda.device_count()-1}")
            device = torch.device('cpu')
        else:
            device = torch.device(f'cuda:{gpu_id}')
            torch.cuda.set_device(gpu_id)
    
        # Get actual hardware GPU ID for display
        actual_gpu_id = gpu_mapping.get(gpu_id, gpu_id) if gpu_mapping else get_actual_gpu_id(gpu_id)
        
        # Move model to the correct device
        ae_model = ae_model.to(device)
        
        results = []

        for slide_path in slides:
            try:
                result = DeepCluster(slide_path, device, config, log_file, ae_model, actual_gpu_id, csv_lock)
                results.append(result)
            except Exception as e:
                print(f"Error processing {slide_path} on GPU {actual_gpu_id}: {e}")
                traceback.print_exc()
        
        return results
        
    except Exception as e:
        print(f"Critical error in GPU worker {gpu_id}: {e}")
        traceback.print_exc()
        return []
  
# Assigning list of input folders to multiple GPUs
def process_all_input_folders_parallel(config, device, log_file, specified_gpu_ids=None):
    """Process all slides using multiple GPUs with proper validation and error handling"""
    try:
        # Set the multiprocessing start method, before creating any locks  
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # Already set, continue
            pass
        
        # Validate CUDA availability first
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to serial processing")
            return process_all_input_folders_serial(config, device, log_file)
        
        # Create the CSV lock AFTER setting spawn method
        csv_lock = mp.Lock()
        
        # Prepare dataset
        if has_no_subfolders(config.input_path) or config.process_all:
            input_folders_list = [config.input_path]          
        else:
            dataset = WSIDataset(config.input_path, config.selected_input_folders, log_file)
            input_folders_list = dataset.slide_files 
    
        if not input_folders_list:
            print(f"\nNo folders found to process in {config.input_path}\n")
            return
    
        # Create GPU mapping with validation
        gpu_mapping = create_gpu_mapping(specified_gpu_ids)
        
        if not gpu_mapping:
            print("No valid GPUs available, falling back to serial processing")
            return process_all_input_folders_serial(config, device, log_file)
        
        # Safe GPU setup
        if specified_gpu_ids:
            # Validate and filter GPU IDs
            total_gpus = torch.cuda.device_count()
            valid_gpu_ids = [gpu_id for gpu_id in specified_gpu_ids if 0 <= gpu_id < total_gpus]
            
            if not valid_gpu_ids:
                print("No valid GPU IDs, using all available GPUs")
                device_ids = list(range(total_gpus))
            else:
                # Set CUDA_VISIBLE_DEVICES to the valid GPUs only
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, valid_gpu_ids))
                # After setting CUDA_VISIBLE_DEVICES, indices are remapped
                device_ids = list(range(len(valid_gpu_ids)))
                #print(f"Set CUDA_VISIBLE_DEVICES to: {valid_gpu_ids}")
        else:
            device_ids = list(range(torch.cuda.device_count()))

        num_gpus = min(len(device_ids), len(input_folders_list))
        
        if num_gpus == 0:
            print("No GPUs available for processing, falling back to serial processing")
            return process_all_input_folders_serial(config, device, log_file)
        
        # Distribute slides across GPUs
        slides_split = [[] for _ in range(num_gpus)]
        
        # Assign one slide to each GPU first
        for i in range(min(num_gpus, len(input_folders_list))):
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
        print('-' * 100)

        # Load autoencoder model on CPU first, then move to GPU in workers
        try:
            ae_model = load_ae_model(config, torch.device('cpu'))
        except Exception as e:
            #print(f"Error loading autoencoder model: {e}")
            return
    
        # Launch processing on each GPU
        processes = []
        
        for pytorch_gpu_id in range(num_gpus):
            # Validation for GPU ID
            if pytorch_gpu_id >= len(device_ids):
                print(f"Skipping invalid GPU ID: {pytorch_gpu_id}")
                continue
                
            process = torch.multiprocessing.Process(
                target=process_input_folder_on_gpu,
                args=(slides_split[pytorch_gpu_id], pytorch_gpu_id, device, config, log_file, ae_model, gpu_mapping, csv_lock)
            )
            processes.append(process)
            process.start()
    
        # Wait for completion
        for process in processes:
            process.join() 

        print(" ")
        print('-' * 100) 
        print("DeepCluster++ completed...!!!")  
        print('-' * 100)
        
        # Clean up any remaining temporary directories
        temp_features_base = os.path.join(config.output_path, 'temp_features')
        if os.path.exists(temp_features_base):
            try: 
                shutil.rmtree(temp_features_base)
                print(f"\nCleaned up all temporary feature directories\n")
            except Exception as e:
                print(f"Warning: Could not clean up temporary feature base directory: {e}") 
                
    except Exception as e:
        print(f"Error during processing: {str(e)}")        
        traceback.print_exc()

# Processing input folders with CPU
def process_all_input_folders_serial(config, device, log_file):    
    try:         
        # Load autoencoder model
        ae_model = load_ae_model(config, device)  
    
        # Prepare dataset
        if has_no_subfolders(config.input_path) or config.process_all:
            input_folders_list = [config.input_path]          
        else:
            dataset = WSIDataset(config.input_path, config.selected_input_folders, log_file)
            input_folders_list = dataset.slide_files 
    
        if not input_folders_list:
            print(f"\nNo folders found to process in {config.input_path}\n")
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
                
        print(' ')
        print('-' * 100 ) 
        print("DeepCluster++ completed...!!!")  
        print('-' * 100 )
        
        # Clean up any remaining temporary directories
        temp_features_base = os.path.join(config.output_path, 'temp_features')        
        if os.path.exists(temp_features_base):
            try: 
                shutil.rmtree(temp_features_base)
                print(f"\nCleaned up all temporary feature directories\n")
            except Exception as e:
                print(f"Warning: Could not clean up temporary feature base directory: {e}")
            
        print(f"\nSuccessfully processed: {successful} input_folders (WSIs)\n")
        if failed > 0:
            print(f"Failed to process: {failed} input_folders (WSIs)\n")
            
    except Exception as e:
        print(f"\nError during processing: {str(e)}")        
        traceback.print_exc()