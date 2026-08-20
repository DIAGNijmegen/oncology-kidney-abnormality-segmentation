#!/usr/bin/env python3
"""
Find the largest sw_batch_size that fits in GPU memory, using the real trained
network and the exact forward-pass path used at inference time
(BatchedNNUNetPredictor._internal_maybe_mirror_and_predict).

Only loads a single fold (folds share the same architecture, so peak memory
per forward pass is identical regardless of ensemble size).

Usage:
    python scripts/benchmark_sw_batch_size.py /path/to/model_dir
    python scripts/benchmark_sw_batch_size.py /path/to/model_dir --fast
    python scripts/benchmark_sw_batch_size.py /path/to/model_dir --max-batch 8
"""

import argparse
import sys

import torch

from kidney_abnormality_segmentation.segmentation.segment_ct_image import get_predictor


def get_num_input_channels(network: torch.nn.Module) -> int:
    for m in network.modules():
        if isinstance(m, (torch.nn.Conv3d, torch.nn.Conv2d)):
            return m.in_channels
    raise RuntimeError("Could not find a Conv layer to infer input channel count from.")


def try_batch_size(predictor, sw_batch_size: int, patch_size, num_input_channels: int):
    """Returns peak memory in GiB if sw_batch_size fits, else None (OOM)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(predictor.device)
    x = None
    try:
        x = torch.rand(
            (sw_batch_size, num_input_channels, *patch_size),
            dtype=torch.float32,
            device=predictor.device,
        )
        predictor._internal_maybe_mirror_and_predict(x)
        return torch.cuda.max_memory_allocated(predictor.device) / (1024 ** 3)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        return None
    finally:
        del x
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_path", help="Path containing nnUNet_results/... (same as passed to inference.run)")
    parser.add_argument("--fast", action="store_true",
                         help="Disable mirroring, matching --fast at inference time.")
    parser.add_argument("--max-batch", type=int, default=8, help="Upper bound to probe up to (default: 8).")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA not available -- this benchmark needs a GPU.")

    print("Loading model (one fold; architecture/memory footprint is the same across folds)...")
    predictor = get_predictor(args.model_path, run_fast=args.fast, run_one_fold=True, sw_batch_size=1)

    predictor.network = predictor.network.to(predictor.device)
    patch_size = predictor.configuration_manager.patch_size
    num_input_channels = get_num_input_channels(predictor.network)
    mirror_variants = 2 ** len(predictor.allowed_mirroring_axes) if predictor.use_mirroring else 1

    print(f"patch_size={patch_size}, num_input_channels={num_input_channels}, "
          f"mirroring={'on' if predictor.use_mirroring else 'off'} "
          f"({mirror_variants} variant(s) -> effective batch = sw_batch_size x {mirror_variants})\n")

    last_ok, last_ok_mem = None, None
    for sw_batch_size in range(1, args.max_batch + 1):
        mem = try_batch_size(predictor, sw_batch_size, patch_size, num_input_channels)
        effective = sw_batch_size * mirror_variants
        if mem is None:
            print(f"sw_batch_size={sw_batch_size:>3} (effective {effective:>3}): OOM")
            break
        print(f"sw_batch_size={sw_batch_size:>3} (effective {effective:>3}): peak {mem:6.2f} GiB")
        last_ok, last_ok_mem = sw_batch_size, mem
    else:
        print(f"\nReached --max-batch={args.max_batch} without OOMing -- raise it to keep probing.")

    print()
    if last_ok is None:
        print("Even sw_batch_size=1 OOMs on this GPU with this patch size / mirroring setting.")
        if not args.fast:
            print("Try --fast (disables mirroring, 8x less effective batch) to see if that alone fits.")
    else:
        print(f"Largest sw_batch_size that fit: {last_ok} (peak {last_ok_mem:.2f} GiB)")
        print("Leave headroom below this -- real CT patches vary in foreground content, and this "
              "doesn't include preprocessing/other processes sharing the GPU.")


if __name__ == "__main__":
    main()
