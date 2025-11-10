import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ViT_B_16_Weights 
from torch.utils.data import Dataset, DataLoader
from Config import Config

class ViTEncoder:
    """Vision Transformer (ViT-B/16) encoder for feature extraction"""
    
    def __init__(self, config: Config, actual_gpu_id = 0):
        """Initialize ViT encoder"""
        self.config = config
        self.device = self._setup_device(actual_gpu_id)
        self.model = self._load_model(self.device)
        self.transform = self._get_transform()
        self.feature_dim = self._get_feature_dim() 
    
    def _setup_device(self, actual_gpu_id):
        """Setup device with proper GPU handling"""
        if self.config.device == "cpu":
            device = torch.device('cpu')
        elif actual_gpu_id >= torch.cuda.device_count():
            device = torch.device('cuda:0')
        else:
            device = torch.device(f'cuda:{actual_gpu_id}')         
        return device
    
    def _load_model(self, device):
        """Load and prepare ViT model"""
        model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        # Remove the classification head (keep only the encoder)
        model.heads = nn.Identity()
        model.eval()
        return model.to(device)
    
    def _get_transform(self):
        """Get image preprocessing transforms"""
        return transforms.Compose([
            transforms.Resize(224), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])
    
    def _get_feature_dim(self):
        """Get feature dimension"""
        return 768
    
    def get_encoder_name(self):
        """Get encoder name"""
        return "vit_b_16"
    
    def extract_features(self, batch_tensor):
        """Extract features from a batch of images using encoder        
        Args:
            batch_tensor: Input batch of images [batch_size, channels, height, width]            
        Returns:
            numpy array of features [batch_size, feature_dim]
        """
        try:
            batch_tensor = batch_tensor.to(self.device)
            
            with torch.no_grad():
                features = self.model(batch_tensor)
                # ViT already outputs 2D features: [batch_size, feature_dim]
                # No need for additional reshaping
                
            return features.cpu().numpy()
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"GPU OOM error during feature extraction: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        except Exception as e:
            print(f"Error during feature extraction: {e}")
            raise