# ====================================
# Main.py
# ====================================
import os
import warnings
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
from Config import Config
from Processing_serial import process_input_folder, process_all_input_folders 

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN 

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