import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from pathlib import Path 
from Config import Config
from . import Auto_encoder

model_path = Path(__file__).parent / "AE_CRC.pth"

class AECRCEncoder:
    """AutoEncoder encoder for feature extraction"""
    
    def __init__(self, config: Config, actual_gpu_id = 0):
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
        """Load and prepare AutoEncoder model"""
        try: 
            if not model_path.exists():
                raise FileNotFoundError(f"Model weights not found at {model_path}")
            
            model = Auto_encoder.AutoEncoder()
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            return model.to(device)
            
        except FileNotFoundError as e:
            print(f"File error loading autoencoder: {e}")
            raise
        except Exception as e:
            print(f"Error loading autoencoder model: {e}")
            raise
    
    def _get_transform(self):
        """Get image preprocessing transforms"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    
    def _get_feature_dim(self):
        """Get feature dimension"""
        return 512
    
    def get_encoder_name(self):
        """Get encoder name"""
        return "autoencoder"
    
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
                # Extract encoder features (bottleneck representation)
                if self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        features = self.model.encoder(batch_tensor)
                        features_gap = F.adaptive_avg_pool2d(features, (1, 1))
                        features_flattened = features_gap.view(features_gap.size(0), -1)
                else:
                    features = self.model.encoder(batch_tensor)
                    features_gap = F.adaptive_avg_pool2d(features, (1, 1))
                    features_flattened = features_gap.view(features_gap.size(0), -1)
                
                # Ensure features are 2D: [batch_size, feature_dim]
                if len(features_flattened.shape) > 2:
                    features_flattened = features_flattened.view(features_flattened.size(0), -1)
            
            return features_flattened.cpu().numpy()
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"GPU OOM error during feature extraction: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        except Exception as e:
            print(f"Error during feature extraction: {e}")
            raise