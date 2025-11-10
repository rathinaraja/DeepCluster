import os, torch
from torchvision import transforms
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import hf_hub_download
from Config import Config

class CONCHEncoder:
    """
    CONCH image-tower encoder via HF Hub.
    Config:
      - Config.conch_repo_id: e.g. "MahmoodLab/CONCH" (set by you)
      - Config.conch_filename: e.g. "pytorch_model.bin" (fallback load)
    """

    def __init__(self, config: Config, actual_gpu_id = 0):
        self.config = config
        self.device = self._setup_device(actual_gpu_id)
        self.model, self.transform = self._build_model_and_transform()
        self.feature_dim = getattr(self.model, "num_features", 768)

    def _setup_device(self, gid):
        if self.config.device == "cpu" or not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(f"cuda:{gid if gid < torch.cuda.device_count() else 0}")

    def _build_model_and_transform(self):
        repo_id = getattr(Config, "conch_repo_id", "")
        filename = getattr(Config, "conch_filename", "pytorch_model.bin")

        # 1) Try direct timm HF-hub loading (if the repo registers a timm entry)
        if repo_id:
            try:
                m = timm.create_model(
                    f"hf-hub:{repo_id}",
                    pretrained=True,
                    num_classes=0,
                    dynamic_img_size=True
                ).to(self.device).eval()
                cfg = resolve_data_config(m.pretrained_cfg, model=m)
                return m, create_transform(**cfg)
            except Exception as e:
                print(f"[CONCH] Direct timm HF load failed: {e}. Falling back to manual checkpoint...")

        # 2) Fallback: download a raw checkpoint and load on a compatible backbone
        # Pick an architecture consistent with your checkpoint (adjust if needed):
        # Many CONCH releases use EVA/ViT-like backbones; vit_large_patch16_224 is a safe default.
        m = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0).to(self.device).eval()

        if repo_id:
            ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
        else:
            raise RuntimeError("Set Config.conch_repo_id to a valid HF repo id for CONCH.")

        state = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in state: state = state["state_dict"]
        state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
        m.load_state_dict(state, strict=False)

        tfm = transforms.Compose([
            transforms.Resize(224), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
        ])
        return m, tfm

    def get_encoder_name(self): return "conch"

    @torch.no_grad()
    def extract_features(self, batch_tensor: torch.Tensor):
        """Extract features from a batch of images using encoder        
        Args:
            batch_tensor: Input batch of images [batch_size, channels, height, width]            
        Returns:
            numpy array of features [batch_size, feature_dim]
        """
        x = batch_tensor.to(self.device, non_blocking=True)
        z = self.model(x)
        if z.ndim > 2: z = z.flatten(1)
        return z.cpu().numpy()
