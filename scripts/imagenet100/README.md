# ImageNet-100

The dataset configured in [ImageNet100.yaml](../../conf/task/ImageNet100.yaml) downloads automatically on first run. Download the frozen ViT-5 backbone checkpoint from the repository root before launching an experiment:

```bash
mkdir -p models/backbones/pretrained/vit5/checkpoints
wget -P models/backbones/pretrained/vit5/checkpoints \
  https://huggingface.co/FengWang3211/ViT-5/resolve/main/vit5_small_patch16_224.pth
```

Run a model's script, for example `bash scripts/imagenet100/cmm.sh`. Ablations are in `ablations/`; `ctm_memory_matched.sh` is CTM-Ext.
