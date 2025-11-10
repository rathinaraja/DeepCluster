import torch
import torch.nn as nn
from torchvision import models, transforms
from Config import Config

class ResNet50_1024Encoder:
    """ResNet50 extracting 1024-d features from 3rd layer"""

    def __init__(self, config: Config, actual_gpu_id = 0):
        self.config = config
        self.device = self._setup_device(actual_gpu_id)
        self.model = self._load_model().to(self.device)
        self.model.eval()
        self.transform = self._get_transform()
        self.feature_dim = 1024

    def _setup_device(self, actual_gpu_id):
        if self.config.device == "cpu":
            return torch.device("cpu")
        if actual_gpu_id >= torch.cuda.device_count():
            return torch.device("cuda:0")
        return torch.device(f"cuda:{actual_gpu_id}")

    def _load_model(self):
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Use only layers up to layer3 (exclude layer4)
        model = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        return model

    def _get_transform(self):
        return transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def get_encoder_name(self):
        return "resnet50_1024"

    @torch.no_grad()
    def extract_features(self, batch_tensor):
        """Extract features from a batch of images using encoder        
        Args:
            batch_tensor: Input batch of images [batch_size, channels, height, width]            
        Returns:
            numpy array of features [batch_size, feature_dim]
        """
        batch_tensor = batch_tensor.to(self.device)
        features = self.model(batch_tensor)
        features = features.view(features.size(0), -1)
        return features.cpu().numpy()
