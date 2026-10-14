# Tests

Run the tests that do not require external datasets or pretrained checkpoints:

```bash
uv run python -m pytest tests/ --ignore=tests/test_vit_checkpoint.py
```

CUDA-specific tests skip when a GPU is unavailable. The analysis tests use synthetic runs and check final-epoch selection, training-seed identity, sample standard deviations, attention aggregation, and plotting.

The ViT integration tests additionally require the [ViT-5 checkpoint](../scripts/imagenet100/README.md) and access to the ImageNet-100 dataset (or its local Hugging Face cache):

```bash
uv run python -m pytest tests/test_vit_checkpoint.py
```
