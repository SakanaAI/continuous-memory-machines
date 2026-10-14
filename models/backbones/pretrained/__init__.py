import sys
import warnings
from pathlib import Path

_VIT5_CODE_DIR = str(Path(__file__).parent / "vit5" / "code")
if _VIT5_CODE_DIR not in sys.path:
    sys.path.insert(0, _VIT5_CODE_DIR)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"timm\.models\.")
    from models_vit5 import vit5_small as vit5

VIT5_CKPT_DIR = Path(__file__).parent / "vit5" / "checkpoints"
