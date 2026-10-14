import torch
import torch.nn as nn

from models.backbones.pretrained import vit5, VIT5_CKPT_DIR

_VIT5_CKPT = "vit5_small_patch16_224.pth"

class ViT5Backbone(nn.Module):
    def __init__(self, img_size: int = 224, layer: int = -2, output_type: str = "vector", pretrained: bool = True):
        super().__init__()

        self.output_type = output_type
        self.vit = vit5(img_size=img_size, num_classes=1000)
        self._features = None

        self.num_patches = self.vit.patch_embed.num_patches # 196
        self.grid_size = int(self.num_patches ** 0.5) # 14
        self.d_token = self.vit.embed_dim # 384

        if pretrained:
            ckpt_path = VIT5_CKPT_DIR / _VIT5_CKPT
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                self.vit.load_state_dict(ckpt["model"], strict=False)
            else:
                raise FileNotFoundError(f"ViT5 checkpoint not found at {ckpt_path}.")

        if output_type in ("vector", "patches"):
            self.vit.blocks[layer].register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self._features = output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.vit(x) # (B, C, H, W) -> (B, num_classes)
        if self.output_type == "class":
            return logits
        elif self.output_type == "patches":
            return self._features[:, 1 : 1 + self.num_patches] # (B, 1+N+R, d_token) -> (B, N, d_token), drop CLS and registers
        elif self.output_type == "vector":
            return self._features[:, 0] # (B, 1+N+R, d_token) -> (B, d_token), CLS token
        else:
            raise ValueError(f"Unknown output_type: {self.output_type}")


class LinearBackbone(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.proj = nn.LazyLinear(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)





def _conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)


def _conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNetBackbone(nn.Module):
    """ResNet backbone matching the CTM paper's configuration.

    Initial conv is kernel=3, stride=1 (not the standard kernel=7, stride=2) to
    preserve spatial resolution on small inputs like mazes.
    """

    _CONFIGS = {
        "resnet18": ([2, 2, 2, 2], _BasicBlock),
        "resnet34": ([3, 4, 6, 3], _BasicBlock),
    }

    def __init__(self, in_channels: int = 3, num_blocks: int = 2, stride: int = 2, arch: str = "resnet34", do_initial_pool: bool = True, output_type: str = "patches"):
        super().__init__()
        assert arch in self._CONFIGS, f"Unknown arch: {arch}. Choose from {list(self._CONFIGS)}"
        assert output_type in ("patches", "vector"), f"Unknown output_type: {output_type}"
        layer_counts, block = self._CONFIGS[arch]

        self.inplanes = 64
        self.output_type = output_type

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False) if in_channels in (1, 3) else nn.LazyConv2d(64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if do_initial_pool else nn.Identity()

        self.layers = nn.ModuleList()
        channels = [64, 128, 256, 512]
        for i in range(min(num_blocks, 4)):
            s = 1 if i == 0 else stride
            self.layers.append(self._make_layer(block, channels[i], layer_counts[i], stride=s))

        self.d_token = channels[min(num_blocks, 4) - 1] * block.expansion

    def _make_layer(self, block, planes, n_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(_conv1x1(self.inplanes, planes * block.expansion, stride), nn.BatchNorm2d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, n_blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool(self.stem(x))
        for layer in self.layers:
            x = layer(x)
        if self.output_type == "patches":
            return x.flatten(2).transpose(1, 2) # (B, C, H', W') -> (B, N, C)
        return x.mean(dim=(2, 3)) # (B, C, H', W') -> (B, C)
