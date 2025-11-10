import os
import torch
import torch.nn as nn
from torchvision import transforms
import timm
from timm.layers import to_2tuple
from Config import Config

# ---- ConvStem (CTRANSPath) with timm/Swin compatibility ----
class ConvStem(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        strict_img_size=None,   # timm passes this
        **kwargs
    ):
        super().__init__()
        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.strict_img_size = True if strict_img_size is None else bool(strict_img_size)

        stem = []
        input_dim, output_dim = in_chans, embed_dim // 8
        for _ in range(2):
            stem += [
                nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True),
            ]
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        if isinstance(norm_layer, type) and issubclass(norm_layer, nn.BatchNorm2d):
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.strict_img_size:
            assert (H, W) == self.img_size, f"Input image size ({H}x{W}) != ({self.img_size[0]}x{self.img_size[1]})"
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

# ---- helpers ----
def _unwrap(sd: dict) -> dict:
    for k in ("state_dict", "model", "net", "ema_model"):
        if k in sd and isinstance(sd[k], dict):
            return sd[k]
    return sd

def _strip_prefix(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        nk = k
        for pref in ("module.", "backbone.", "model."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        out[nk] = v
    return out

def _sanitize(sd: dict) -> dict:
    drop_prefixes = (
        "head.", "fc.", "classifier.", "logit.", "cls_head.",
        # registered buffers in Swin we should not try to load
        "layers.0.blocks.0.attn.relative_position_index",
        "layers.0.blocks.1.attn.relative_position_index",
        "layers.0.blocks.1.attn_mask",
        "layers.1.blocks.0.attn.relative_position_index",
        "layers.1.blocks.1.attn.relative_position_index",
        "layers.1.blocks.1.attn_mask",
        "layers.2.blocks.0.attn.relative_position_index",
        "layers.2.blocks.1.attn.relative_position_index",
        "layers.2.blocks.1.attn_mask",
        "layers.2.blocks.2.attn.relative_position_index",
        "layers.2.blocks.3.attn.relative_position_index",
        "layers.2.blocks.3.attn_mask",
        "layers.2.blocks.4.attn.relative_position_index",
        "layers.2.blocks.5.attn.relative_position_index",
        "layers.2.blocks.5.attn_mask",
        "layers.3.blocks.0.attn.relative_position_index",
        "layers.3.blocks.1.attn.relative_position_index",
    )
    out = {}
    for k, v in sd.items():
        if any(k.startswith(p) for p in drop_prefixes):
            continue
        # ignore patch_embed.norm.* — our ConvStem has its own normalization
        if k.startswith("patch_embed.norm."):
            continue
        out[k] = v
    return out

def _infer_arch_from_state(sd: dict) -> str:
    """
    Infer Swin size from checkpoint by:
      - embed_dim from patch stem channels (conv stem final out = embed_dim)
      - depth of stage 3 (layers.2.*) — Tiny: 6, Small: 18, Base: 18, Large: 18
      - embed_dim 96 => Tiny/Small; 128 => Base; 192 => Large
    """
    # Try to read ConvStem channels from earliest conv weight
    # patch_embed.proj.0.weight has shape [Cout, Cin, 3, 3]; Cout should be embed_dim//8
    cout = None
    for key in ("patch_embed.proj.0.weight", "patch_embed.0.weight", "patch_embed.conv1.weight"):
        if key in sd:
            cout = sd[key].shape[0]
            break
    embed_dim = None
    if cout is not None:
        embed_dim = cout * 8  # ConvStem first conv channels = embed_dim // 8

    # Count how many blocks in stage 3 (layers.2.blocks.*)
    max_idx_stage3 = -1
    for k in sd.keys():
        if k.startswith("layers.2.blocks."):
            try:
                idx = int(k.split(".")[3])
                if idx > max_idx_stage3:
                    max_idx_stage3 = idx
            except Exception:
                pass
    depth_stage3 = max_idx_stage3 + 1 if max_idx_stage3 >= 0 else None

    # Decide
    if embed_dim in (96, 128, 192):
        if embed_dim == 96:
            # depth 6 => Tiny, 18 => Small
            if depth_stage3 == 6:
                return "swin_tiny_patch4_window7_224"
            if depth_stage3 == 18:
                return "swin_small_patch4_window7_224"
            # fallback: prefer Tiny for 96 if unsure
            return "swin_tiny_patch4_window7_224"
        if embed_dim == 128:
            return "swin_base_patch4_window7_224"
        if embed_dim == 192:
            return "swin_large_patch4_window7_224"

    # Fallback: Tiny is the most common for CTransPath
    return "swin_tiny_patch4_window7_224"

def _build_swin(arch: str, device: torch.device) -> nn.Module:
    m = timm.create_model(
        arch,
        pretrained=False,
        num_classes=0,
        embed_layer=ConvStem
    ).to(device).eval()
    if hasattr(m, "head") and not isinstance(m.head, nn.Identity):
        m.head = nn.Identity()
    return m

class CTransPathEncoder:
    def __init__(self, config: Config, actual_gpu_id = 0):
        self.config = config
        self.device = self._setup_device(actual_gpu_id)

        ckpt_path = getattr(Config, "ctrans_local_ckpt", "")
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise RuntimeError(f"CTransPath checkpoint not found: {ckpt_path}")

        raw = torch.load(ckpt_path, map_location="cpu")
        sd = _sanitize(_strip_prefix(_unwrap(raw)))

        # Force or infer the correct Swin
        forced = getattr(Config, "ctrans_force_arch", "").strip()
        if forced:
            arch = forced
        else:
            arch = _infer_arch_from_state(sd)
        print(f"--- DIAGNOSIS: Attempting to build model with arch: '{arch}' ---") 
        # Build and load
        model = _build_swin(arch, self.device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
   
        print(f"[CTransPath] Selected backbone: {arch}")
        print(f"[CTransPath] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

        self.model = model
        self.transform = transforms.Compose([
            transforms.Resize(224), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])
        self.feature_dim = getattr(self.model, "num_features", 384)

    def _setup_device(self, gid):
        if self.config.device == "cpu" or not torch.cuda.is_available():
            return torch.device("cpu")
        if gid >= torch.cuda.device_count():
            gid = 0
        return torch.device(f"cuda:{gid}")

    def get_encoder_name(self):
        return "ctranspath"

    @torch.no_grad()
    def extract_features(self, batch_tensor):
        """Extract features from a batch of images using encoder        
        Args:
            batch_tensor: Input batch of images [batch_size, channels, height, width]            
        Returns:
            numpy array of features [batch_size, feature_dim]
        """
        x = batch_tensor.to(self.device, non_blocking=True)
        z = self.model(x)
        if z.ndim > 2:
            z = z.flatten(1)
        return z.cpu().numpy()
