# ViT-5 Source Files

These files were pulled from the [ViT-5 GitHub repo](https://github.com/wangf3014/ViT-5):

```bash
mkdir -p models/backbones/pretrained/vit5/code
wget -P models/backbones/pretrained/vit5/code \
  https://raw.githubusercontent.com/wangf3014/ViT-5/main/models_vit5.py \
  https://raw.githubusercontent.com/wangf3014/ViT-5/main/rope.py
```

`rope.py` has been patched to replace a hardcoded `.cuda()` call with `device=self.freqs.device` for CPU compatibility.
