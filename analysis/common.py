"""Paths, experiment manifests, and CSV output shared by the paper analyses."""

import csv
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = Path(__file__).resolve().parent / "configs"
MODEL_ORDER = [
    "Transformer",
    "RNN",
    "LSTM",
    "FSRNN",
    "GDN",
    "DNC",
    "RMC",
    "LSAM",
    "NAMTM",
    "RMT",
    "CTM",
    "LSTM_CMM",
    "CTM_Matched",
    "CTMTransformer",
    "CMM",
]
DISPLAY_NAMES = {
    **{model: model for model in MODEL_ORDER},
    "FSRNN": "FS-LSTM",
    "NAMTM": "NAM-TM",
    "LSTM_CMM": "CMM-LSTM",
    "CTM_Matched": "CTM-Ext.",
    "CTMTransformer": "CMM-STM",
    "GDNStateTracking": "GDN",
}


def load_config(path, runs_root=ROOT):
    """Resolve recorded output paths against a checkout or archived-run root."""

    def resolve(value):
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if isinstance(value, str) and value.startswith("outputs/"):
            return str(Path(runs_root).resolve() / value)
        return value

    with Path(path).open() as handle:
        return resolve(yaml.safe_load(handle))


def seed_dirs(base_dir):
    """Accept a single Hydra run or a multirun directory with numeric children."""
    base_dir = Path(base_dir)
    if (base_dir / ".hydra/config.yaml").is_file():
        return [base_dir]
    if not base_dir.is_dir():
        return []
    return sorted(
        (path for path in base_dir.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )


def write_csv(path, rows):
    """Write atomically, retaining full numeric precision."""
    path = Path(path)
    if not rows:
        raise ValueError(f"No results to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
