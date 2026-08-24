#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import itertools
import time

import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.helpers import empty_cache
from tqdm import tqdm


class BatchedNNUNetPredictor(nnUNetPredictor):
    """
    nnUNetPredictor that batches GPU work along two axes instead of nnU-Net's stock
    one-patch-at-a-time, one-mirror-variant-at-a-time loops:

      1. sw_batch_size sliding-window patches are stacked into one network dispatch
         (_internal_predict_sliding_window_return_logits override).
      2. All TTA mirror variants (up to 8, when mirroring is enabled) are stacked into
         that same dispatch instead of being looped over sequentially
         (_internal_maybe_mirror_and_predict override).

    Effective network batch size = sw_batch_size * num_mirror_variants (up to 8x when
    mirroring is on, the default unless --fast is passed). Peak activation memory
    scales with this product, not with sw_batch_size alone -- tune sw_batch_size down
    if enabling mirroring pushes you into OOM territory. Note nnU-Net's built-in
    OOM fallback (retrying with do_on_device=False) does NOT help here: it only
    changes where results tensors live, not the network's forward batch size, so an
    OOM from too-large a batch will recur identically on retry.

    Drops the upstream producer-thread prefetch, which overlapped host->device
    transfer with compute -- irrelevant here since this repo always runs with
    perform_everything_on_device=True, so the whole volume is already GPU-resident
    before slicing.
    """

    def __init__(self, *args, sw_batch_size: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        if sw_batch_size < 1:
            raise ValueError(f"sw_batch_size must be >= 1, got {sw_batch_size}")
        self.sw_batch_size = sw_batch_size

    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        if mirror_axes is None:
            return self.network(x)

        assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'
        mirror_axes = [m + 2 for m in mirror_axes]
        # () first = unflipped/identity variant, matching upstream's "prediction = self.network(x)" seed
        axes_combinations = [()] + [
            c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
        ]

        b = x.shape[0]
        stacked_input = torch.cat(
            [torch.flip(x, axes) if axes else x for axes in axes_combinations], dim=0
        )
        stacked_output = self.network(stacked_input)

        prediction = stacked_output[:b].clone()
        for i, axes in enumerate(axes_combinations[1:], start=1):
            variant = stacked_output[i * b:(i + 1) * b]
            prediction += torch.flip(variant, axes)
        prediction /= len(axes_combinations)
        return prediction

    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(
        self,
        data: torch.Tensor,
        slicers: list[tuple[slice, ...]],
        do_on_device: bool = True,
    ) -> torch.Tensor:
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')
        t_start = time.perf_counter()

        try:
            empty_cache(self.device)
            data = data.to(results_device)

            predicted_logits = torch.zeros(
                (self.label_manager.num_segmentation_heads, *data.shape[1:]),
                dtype=torch.half, device=results_device,
            )
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            gaussian = compute_gaussian(
                tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                value_scaling_factor=10, device=results_device,
            ) if self.use_gaussian else 1

            chunks = [slicers[i:i + self.sw_batch_size] for i in range(0, len(slicers), self.sw_batch_size)]

            with tqdm(total=len(slicers), disable=not self.allow_tqdm) as pbar:
                for chunk in chunks:
                    workon = torch.stack([data[sl] for sl in chunk], dim=0).to(self.device)
                    prediction = self._internal_maybe_mirror_and_predict(workon).to(results_device)

                    if self.use_gaussian:
                        prediction *= gaussian
                    for j, sl in enumerate(chunk):
                        predicted_logits[sl] += prediction[j]
                        n_predictions[sl[1:]] += gaussian
                    pbar.update(len(chunk))

            torch.div(predicted_logits, n_predictions, out=predicted_logits)
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError(
                    'Encountered inf in predicted array. Aborting... If this problem persists, '
                    'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                    'predicted_logits to fp32'
                )
            print(f"[timing] sliding-window inference (this fold): {time.perf_counter() - t_start:.2f}s")
            return predicted_logits
        except Exception:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise
