"""Profile the paper models on identical complete Copy-100 sequences.

Each model runs in a fresh process on one GPU. Results include the input/output
projections, use eager float32 execution, and exclude optimizer updates. Forward
measurements disable autograd; forward+backward uses the sum of all outputs.
"""

import argparse
import csv
import json
import multiprocessing as mp
import platform
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from analysis.compute.benchmark import (
    D_INPUT,
    FullSequenceModel,
    MODEL_SPECS,
    _build_transformer,
    _count_params,
    _materialise_lazy,
    find_param_matched_d_model,
    profile_sequence,
)
from models.cells.ctm import ContinuousThoughtMachine
from models.cells.ctm_transformer import CTMTransformerCell
from models.rmt import RecurrentMemoryTransformer

ORDER = [
    "Transformer",
    "RNN",
    "LSTM",
    "FSRNN",
    "GDNStateTracking",
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
SPECS = {name: (cls, kwargs, heads) for name, cls, kwargs, heads in MODEL_SPECS}
SPECS["CTM_Matched"] = (
    ContinuousThoughtMachine,
    lambda d: dict(
        d_input=D_INPUT,
        d_model=d,
        memory_length=138,
        dropout=0.0,
        synapse_type="mlp",
        synapse_depth=8,
    ),
    1,
)
SPECS["CTMTransformer"] = (
    CTMTransformerCell,
    lambda d: dict(
        d_input=D_INPUT,
        d_model=d,
        memory_length=138,
        dropout=0.0,
        synapse_type="mlp",
        synapse_depth=8,
        memory_transformer_layers=1,
        num_heads=4,
        memory_update_gate=False,
    ),
    4,
)


class NativeSequenceModel(FullSequenceModel):
    def initial_state(self, batch_size, device):
        if isinstance(self.core, RecurrentMemoryTransformer):
            return self.core.get_initial_state(batch_size)
        return ()

    def forward(self, inputs):
        features = self.backbone(inputs)
        hidden, _ = self.core(
            features, state=self.initial_state(inputs.shape[1], inputs.device)
        )
        return self.output_proj(hidden)


def build_native(name, width):
    if name == "Transformer":
        core = _build_transformer(width, 256)
    else:
        core = RecurrentMemoryTransformer(
            d_input=D_INPUT,
            d_model=width,
            num_heads=4,
            num_layers=2,
            segment_length=10,
            num_mem_tokens=128,
            ff_multiplier=2,
            dropout=0.0,
            attention_dropout=0.0,
            pre_lnorm=False,
        )
    return NativeSequenceModel(core, width, recurrent=False)


def build_model(name, max_params):
    if name in ("Transformer", "RMT"):
        low, high, best = 1, 256, None
        while low <= high:
            mid = (low + high) // 2
            candidate = build_native(name, mid * 4)
            count = _count_params(candidate)
            del candidate
            if count <= max_params:
                best, low = mid * 4, mid + 1
            else:
                high = mid - 1
        if best is None:
            raise ValueError(f"No {name} model fits the parameter budget")
        return build_native(name, best), best
    cls, kwargs, heads = SPECS[name]
    width, _ = find_param_matched_d_model(name, cls, kwargs, heads, max_params)
    if width is None:
        raise ValueError(f"No {name} model fits the parameter budget")
    core = cls(**kwargs(width))
    _materialise_lazy(core)
    return FullSequenceModel(core, width, recurrent=True), width


def worker(name, args, destination):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    torch.cuda.set_device(0)
    model, width = build_model(name, args.max_params)
    result = profile_sequence(
        model,
        args.batch_size,
        200,
        "cuda",
        n_warmup=args.n_warmup,
        n_measure=args.n_measure,
    )
    state = model.initial_state(1, "cuda")
    result["state_mem_kb"] = sum(t.numel() * t.element_size() for t in state) / 1024
    result = dict(
        model=name,
        d_model=width,
        batch_size=args.batch_size,
        sequence_length=200,
        **result,
    )
    result["fwd_per_step_ms"] = result["fwd_ms"] / 200
    result["fwd_bwd_per_step_ms"] = result["fwd_bwd_ms"] / 200
    assert result["params"] <= args.max_params
    assert result["peak_mem_mb"] > 0
    destination.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=ORDER, default=ORDER)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-params", type=int, default=500000)
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-measure", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/analysis/compute")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        vars(args),
        output_dir=str(args.output_dir),
        python=platform.python_version(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        dtype="float32",
        compile=False,
        sequence_length=200,
        input_features=64,
        backward_objective="sum of all outputs",
        optimizer_step=False,
        timing="synchronized wall-clock mean",
        isolated_process_per_model=True,
        parameter_count="complete model including input and output projections",
        state_memory="per example; initial persistent state; Transformer has none",
        peak_memory="peak allocated CUDA memory for full forward+backward",
        gdn_display_name="GDN uses GDNStateTracking",
    )
    (args.output_dir / "protocol.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    context = mp.get_context("spawn")
    rows = []
    for name in args.models:
        print(f"Profiling {name}...", flush=True)
        destination = args.output_dir / f"{name}.json"
        process = context.Process(target=worker, args=(name, args, destination))
        process.start()
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"{name} benchmark exited with code {process.exitcode}")
        row = json.loads(destination.read_text())
        rows.append(row)
        with (args.output_dir / "profile_copy100_full_sequence.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"  d={row['d_model']} params={row['params']:,} "
            f"fwd={row['fwd_ms']:.3f} ms fwd+bwd={row['fwd_bwd_ms']:.3f} ms "
            f"peak={row['peak_mem_mb']:.2f} MiB",
            flush=True,
        )


if __name__ == "__main__":
    main()
