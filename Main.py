import os, csv, argparse, warnings 
import numpy as np
import torch 

from Config import Config
from DeepCluster_framework import process_all_input_folders_parallel
from DeepCluster_framework import process_all_input_folders_serial
from Logging import create_log_file

warnings.filterwarnings("ignore")  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'       # Suppress TF warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'      # Disable oneDNN 

def str2bool(s: str) -> bool:
    if s.lower() in {'yes', 'true', 't', '1'}:
        return True
    if s.lower() in {'no', 'false', 'f', '0'}:
        return False
    raise argparse.ArgumentTypeError('Expected a boolean value.')


def main(): 
    
    parser = argparse.ArgumentParser(description='Cluster and sample WSI images from subfolders with parallel processing support') 
    
    # Required arguments
    parser.add_argument('--input_path', required=True, help='Path to the input folders (WSIs) containing subfolders with images')
    parser.add_argument('--selected_input_folders', type=str, default=None, help='Comma-separated input folder (WSI) names to process. If None, all folders (WSI) will be processed.')
    parser.add_argument('--sub_folders', type=str, default=None, help='Comma-separated sub-folder names in the input folders. If None, all sub_folders will be processed.')
    parser.add_argument('--process_all', type=str2bool, default=False, help='True will switch to process all images in all the input folders. False by default turns it off.')   
    parser.add_argument('--output_path', required=True, help='Path to the output folder')   

    # Parallel processing arguments
    parser.add_argument('--gpu_ids', type=str, default=None, help='Comma-separated GPU IDs for parallel processing (e.g., "0,1,2,3")') 
    parser.add_argument('--device', type=str, default='all_gpus', choices=['all_gpus', 'cpu'], help='Device to use all gpus or cpu or specified gpus') 
    
    # Optional arguments
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for feature extraction (default: 128)') 
    parser.add_argument('--dim_reduce', type=int, default=256, help='Dimensionality reduction before clustering (default: 256)') 
    parser.add_argument('--distance_groups', type=int, default=5, help='Number of distance groups for sampling (default: 5)')
    parser.add_argument('--sample_percentage', type=float, default=0.20,  help='Percentage of images to sample from each group (default: 0.20)') 
    parser.add_argument('--model', type=str,  default="AE_CRC.pth", help='Path to the model file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)') 

    # Enabling/disabling output folders
    parser.add_argument('--store_features', type=str2bool, default=False, help='Store features and cluster assignment files permanently (default: False)')
    parser.add_argument('--store_clusters', type=str2bool, default=False, help='Store clusters folder with clustered images (default: False)')
    parser.add_argument('--store_plots', type=str2bool, default=False, help='Store plots folder with visualization plots (default: False)')
    parser.add_argument('--store_samples', type=str2bool, default=False, help='Store samples folder with sampled images (default: False)')
    parser.add_argument('--store_samples_group_wise', type=str2bool, default=False, help='Store samples in group-wise folders within clusters (default: False)')
    
    args = parser.parse_args()  
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)  

    # Create config object
    config = Config(
        input_path = args.input_path,
        selected_input_folders = args.selected_input_folders, 
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
    
    # File to write metrics
    log_file = args.output_path+'/Summary.csv' 
    create_log_file(config, log_file)
    
    # Set up GPUs   
    device = None
    if args.gpu_ids is not None:
        print("-----------------------------------------------------")
        print("Parallel Processing Mode with GPUs")
        print("-----------------------------------------------------")    
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
        if torch.cuda.is_available() and gpu_ids:
            print(f"GPU IDs: {','.join(map(str, gpu_ids))}")
            for gpu_id in gpu_ids:
                print(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            print("No CUDA GPUs available.")   
        print("-----------------------------------------------------")
    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        # Remove csv_lock parameter - it's created inside the function now
        process_all_input_folders_parallel(config, device, log_file, gpu_ids)
        
    # All GPUs     
    elif args.device == "all_gpus":        
        print("-----------------------------------------------------")
        print("Parallel Processing Mode with GPUs")
        print("-----------------------------------------------------")    
        gpu_ids = list(range(torch.cuda.device_count()))  # Use all GPUs
        if torch.cuda.is_available() and gpu_ids:
            print(f"GPU IDs: {','.join(map(str, gpu_ids))}")
            for gpu_id in gpu_ids:
                print(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            print("No CUDA GPUs available.")   
        print("-----------------------------------------------------")
    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Remove csv_lock parameter - it's created inside the function now
        process_all_input_folders_parallel(config, device, log_file, gpu_ids)

    # Only CPU, Single-threaded mode (original behavior)
    else:  
        device = torch.device('cpu')         
        print("-----------------------------------------------------")
        print("Single-threaded Processing Mode")
        print("-----------------------------------------------------") 
        print(f"Using device: {device}")
        print("-----------------------------------------------------")
        
        # Process the subfolders using single-threaded mode
        # Remove csv_lock parameter - serial function doesn't need it
        process_all_input_folders_serial(config, device, log_file)

if __name__ == "__main__":
    main()