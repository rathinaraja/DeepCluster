import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNet18_Weights 
from torch.utils.data import Dataset, DataLoader

from Config import Config

class ResNet18Encoder:
    """ResNet18 encoder for feature extraction"""
    
    def __init__(self, config: Config, actual_gpu_id = 0):
        """Initialize ResNet18 encoder"""
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
        """Load and prepare ResNet18 model"""
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model = nn.Sequential(*list(model.children())[:-1])  # Remove final layer
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
        return 512
    
    def get_encoder_name(self):
        """Get encoder name"""
        return "resnet18"
    
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
                # Ensure features are 2D: [batch_size, feature_dim]
                if len(features.shape) > 2:
                    features = features.view(features.size(0), -1)
                
            return features.cpu().numpy()
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"GPU OOM error during feature extraction: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        except Exception as e:
            print(f"Error during feature extraction: {e}")
            raise