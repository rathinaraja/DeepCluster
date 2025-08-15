# ====================================
# Dataset.py  
# ==================================== 
import os
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from Config import Config

# Common image extensions used throughout the module
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff') 

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
      