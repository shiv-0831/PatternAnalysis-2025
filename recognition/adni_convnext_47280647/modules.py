import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyCNN(nn.Module):
    """Minimal 2D CNN for binary classification on 1×224×224 inputs."""
    def __init__(self, in_chans: int = 1, num_classes: int = 2, head_dropout: float = 0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_chans, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),   # 112×112
            nn.Conv2d(16, 32, 3, padding=1),        nn.ReLU(inplace=True), nn.MaxPool2d(2),   # 56×56
            nn.Conv2d(32, 64, 3, padding=1),        nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))                                                # 64×1×1
        )
        self.dropout = nn.Dropout(p=head_dropout) if head_dropout > 0 else nn.Identity()
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)               # (B,64,1,1)
        x = x.flatten(1)                   # (B,64)
        x = self.dropout(x)
        logits = self.head(x)              # (B,2)
        probs = F.softmax(logits, dim=1)
        return {"logits": logits, "probs": probs}

def build_model(in_chans: int = 1, num_classes: int = 2, head_dropout: float = 0.0) -> nn.Module:
    return TinyCNN(in_chans=in_chans, num_classes=num_classes, head_dropout=head_dropout)

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
