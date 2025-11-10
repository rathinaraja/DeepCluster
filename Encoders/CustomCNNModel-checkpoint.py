import torch
import torch.nn as nn

class CustomCNNModel(nn.Module):
    """Extract features from a batch of images using encoder        
    Args:
        batch_tensor: Input batch of images [batch_size, channels, height, width]            
    Returns:
        numpy array of features [batch_size, feature_dim]
    """
    
    def __init__(self):
        super(CustomCNNModel, self).__init__()
        
        # Input: 3 x 224 x 224
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)
        
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        
        return x