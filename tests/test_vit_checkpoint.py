import json
import random
from pathlib import Path

import torch
from datasets import load_dataset
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from models.backbones import ViT5Backbone

_CLASSES_JSON = Path(__file__).resolve().parent.parent / "utils" / "imagenet1k_classes.json"
with open(_CLASSES_JSON) as f:
    IMAGENET1K_NAMES = json.load(f)

def test_backbone_output_shape():
    backbone = ViT5Backbone(layer=-2)
    x = torch.randn(2, 3, 224, 224)
    features = backbone(x)
    assert features.shape == (2, 384)

def test_backbone_patches_shape():
    backbone = ViT5Backbone(layer=-2, output_type="patches")
    x = torch.randn(2, 3, 224, 224)
    tokens = backbone(x)
    assert tokens.shape == (2, 196, 384)
    assert backbone.num_patches == 196
    assert backbone.grid_size == 14
    assert backbone.d_token == 384

def test_imagenet100_accuracy():
    backbone = ViT5Backbone(output_type="class")
    backbone.eval()

    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])

    dataset = load_dataset("ilee0022/ImageNet100", split="validation")
    random.seed(42)
    indices = random.sample(range(len(dataset)), 10)

    correct = 0
    for idx in indices:
        sample = dataset[idx]
        image = sample["image"].convert("RGB")
        gt = sample["text"].lower().strip()

        x = transform(image).unsqueeze(0)
        with torch.no_grad():
            logits = backbone(x)

        top5 = [IMAGENET1K_NAMES[i].lower() for i in logits.topk(5, dim=1).indices[0].tolist()]
        correct += any(gt in name or name in gt for name in top5)

    assert correct >= 8
