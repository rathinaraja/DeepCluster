# Encoders/UNIEncoder.py
import os
import torch
from timm import create_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import Config

class UNIEncoder:
    """
    UNI encoder (ViT-L/16 via DINOv2) loaded from HuggingFace through timm.
    Requires: huggingface-cli login (or HF_TOKEN env), repo access granted.
    """

    def __init__(self, config: Config, actual_gpu_id = 0):
        self.config = config
        self.device = self._setup_device(actual_gpu_id)
        # Official timm invocation for UNI (from model card)
        self.model = create_model(
            "hf-hub:MahmoodLab/UNI",
            pretrained=True,
            init_values=1e-5,          # LayerScale params
            dynamic_img_size=True,
            num_classes=0              # return features
        ).to(self.device).eval()

        # Build the exact transform from model’s pretrained cfg
        cfg = resolve_data_config(self.model.pretrained_cfg, model=self.model)
        self.transform = create_transform(**cfg)
        # According to UNI, ViT-L/16 embedding dim = 1024
        self.feature_dim = getattr(self.model, "num_features", 1024)

    def _setup_device(self, gid: int):
        if self.config.device == "cpu" or not torch.cuda.is_available():
            return torch.device("cpu")
        if gid >= torch.cuda.device_count():
            gid = 0
        return torch.device(f"cuda:{gid}")

    def get_encoder_name(self):
        return "uni"

    @torch.no_grad()
    def extract_features(self, batch_tensor: torch.Tensor):
        """Extract features from a batch of images using encoder        
        Args:
            batch_tensor: Input batch of images [batch_size, channels, height, width]            
        Returns:
            numpy array of features [batch_size, feature_dim]
        """
        x = batch_tensor.to(self.device, non_blocking=True)
        feats = self.model(x)         # [B, 1024]
        if feats.ndim > 2:
            feats = feats.flatten(1)
        return feats.cpu().numpy()
