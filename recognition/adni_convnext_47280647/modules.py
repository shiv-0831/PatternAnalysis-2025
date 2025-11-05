import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# ------------------------------
# 1) Minimal CNN
# ------------------------------
class TinyCNN(nn.Module):
    """Minimal 2D CNN for binary classification on 1x224x224 inputs."""
    def __init__(self, in_chans: int = 1, num_classes: int = 2, head_dropout: float = 0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_chans, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 112x112
            nn.Conv2d(16, 32, 3, padding=1),        nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 56x56
            nn.Conv2d(32, 64, 3, padding=1),        nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))                                                # 64x1x1
        )
        self.dropout = nn.Dropout(p=head_dropout) if head_dropout > 0 else nn.Identity()
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        x = self.dropout(x)
        logits = self.head(x)
        probs = F.softmax(logits, dim=1)
        return {"logits": logits, "probs": probs}

# ------------------------------
# 2) ConvNeXt-Lite
#    Differences vs “canonical” ConvNeXt:
#      - depthwise kernel 5 (not 7)
#      - SiLU activation (not GELU)
#      - GroupNorm(1,C) as NCHW LayerNorm
#      - two-step stem downsampling (3x3 s=2 twice)
#      - depths=[2,4,8,2], dims=[80,160,320,640]
# ------------------------------

class StochasticDepth(nn.Module):
    """Per-sample DropPath (stochastic depth)."""
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        # shape: [N, 1, 1, 1] so each sample is dropped/kept whole
        mask = torch.empty(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x * mask / keep

class ChannelNorm(nn.Module):
    """
    LayerNorm for NCHW via GroupNorm(1, C): normalizes each channel with affine params.
    This is a standard trick to avoid permute for channels-last LN.
    """
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.GroupNorm(1, num_channels, eps=eps, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

class NeXtLiteBlock(nn.Module):
    """
    ConvNeXt-inspired block (distinct variant):
      depthwise 5x5 -> ChannelNorm -> 1x1 (4x) -> SiLU -> 1x1 (proj) -> LayerScale -> StochasticDepth -> +res
    """
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init: float = 1e-6):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)  # K=5
        self.norm = ChannelNorm(dim)
        self.pw_expand = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.SiLU()  # different from GELU
        self.pw_proj = nn.Conv2d(4 * dim, dim, kernel_size=1)
        # per-channel layerscale (optional)
        self.alpha = nn.Parameter(torch.full((dim,), layer_scale_init)) if layer_scale_init > 0 else None
        self.drop = StochasticDepth(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw(x)
        x = self.norm(x)
        x = self.pw_expand(x)
        x = self.act(x)
        x = self.pw_proj(x)
        if self.alpha is not None:
            x = x * self.alpha[:, None, None]
        x = self.drop(x)
        return x + residual

class ConvNeXtLite(nn.Module):
    """
    A compact, convnext-inspired classifier for 1x224x224 inputs.
    From-scratch, no external backbones, distinct config (dims/depths/kernels/norm/act).
    """
    def __init__(
        self,
        in_chans: int = 1,
        num_classes: int = 2,
        depths: List[int] = [2, 4, 8, 2],
        dims:   List[int] = [80, 160, 320, 640],
        drop_path_rate: float = 0.15,
        head_dropout: float = 0.2,
        layer_scale_init: float = 1e-6,
    ):
        super().__init__()
        assert len(depths) == 4 and len(dims) == 4

        # Stem: two 3x3 stride-2 downsamples (224 -> 56)
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0]//2, kernel_size=3, stride=2, padding=1),  # 224->112
            ChannelNorm(dims[0]//2),
            nn.SiLU(),
            nn.Conv2d(dims[0]//2, dims[0], kernel_size=3, stride=2, padding=1),   # 112->56
            ChannelNorm(dims[0]),
            nn.SiLU(),
        )

        # Downsample layers between stages: 2x2 stride-2 convs
        self.downsamples = nn.ModuleList([
            nn.Sequential(ChannelNorm(dims[0]), nn.Conv2d(dims[0], dims[1], 2, 2)),
            nn.Sequential(ChannelNorm(dims[1]), nn.Conv2d(dims[1], dims[2], 2, 2)),
            nn.Sequential(ChannelNorm(dims[2]), nn.Conv2d(dims[2], dims[3], 2, 2)),
        ])

        # Stages with progressive stochastic depth
        total_blocks = sum(depths)
        dp_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        cursor = 0
        self.stages = nn.ModuleList()
        for stage_idx in range(4):
            blocks = []
            width = dims[stage_idx]
            for _ in range(depths[stage_idx]):
                blocks.append(NeXtLiteBlock(width, drop_path=dp_rates[cursor], layer_scale_init=layer_scale_init))
                cursor += 1
            self.stages.append(nn.Sequential(*blocks))

        # Head: global avg pool -> LN (via GroupNorm) -> dropout -> linear
        self.head_norm = ChannelNorm(dims[-1])
        self.head_drop = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        self.head_fc   = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages[0](x)
        x = self.downsamples[0](x)
        x = self.stages[1](x)
        x = self.downsamples[1](x)
        x = self.stages[2](x)
        x = self.downsamples[2](x)
        x = self.stages[3](x)
        # global average pooling
        x = x.mean(dim=(2, 3))
        return x

    def forward(self, x: torch.Tensor):
        feats = self.forward_features(x)
        feats = self.head_norm(feats[:, :, None, None]).squeeze(-1).squeeze(-1)
        feats = self.head_drop(feats)
        logits = self.head_fc(feats)
        probs = F.softmax(logits, dim=1)
        return {"logits": logits, "probs": probs}

# ------------------------------
# 3) Factory & utility
# ------------------------------
def build_model(
    name: str = "tiny",
    in_chans: int = 1,
    num_classes: int = 2,
    head_dropout: float = 0.0,
    drop_path_rate: float = 0.15,
    layer_scale_init: float = 1e-6,
) -> nn.Module:
    """
    Backward-compatible factory.
    - name="tiny"          -> TinyCNN (baseline)
    - name="nextlite_tiny" -> ConvNeXtLite
    """
    if name == "tiny":
        return TinyCNN(in_chans=in_chans, num_classes=num_classes, head_dropout=head_dropout)
    elif name == "nextlite_tiny":
        return ConvNeXtLite(
            in_chans=in_chans,
            num_classes=num_classes,
            depths=[2, 4, 8, 2],
            dims=[80, 160, 320, 640],
            drop_path_rate=drop_path_rate,
            head_dropout=head_dropout,
            layer_scale_init=layer_scale_init,
        )
    else:
        raise ValueError(f"Unknown model '{name}' (use 'tiny' or 'nextlite_tiny').")

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
