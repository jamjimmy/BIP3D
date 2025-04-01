from mmengine.model import BaseModel
from bip3d.registry import MODELS
from torch import nn

@MODELS.register_module()
class Slam3rProj(BaseModel):
    def __init__(self, **kwargs):
        self.proj = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),  # [h, w, 4] -> [h, w, 16]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # [h, w, 16] -> [h, w, 32]
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1), # [h, w, 32] -> [h, w, 1]
            nn.ReLU()
        )
    def init_weights(self):
        for layer in self.slam3r_proj:
            if isinstance(layer, nn.Conv2d):
                if layer.kernel_size == (1, 1):  # 1x1卷积，手动0初始化
                    nn.init.zeros_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                else:
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.proj(x)