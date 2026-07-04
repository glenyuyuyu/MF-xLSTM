"""
MF-xLSTM implementation for NeuralHydrology.

This file is intended to replace NeuralHydrology's original single-frequency
``x_lstm.py`` implementation. After replacement and removal of ``xlstm`` from
NeuralHydrology's ``SINGLE_FREQ_MODELS`` registry, the model can be used with
multi-frequency inputs in the same general workflow as MTS-LSTM.

The implementation follows the MF-LSTM input-handling idea of Acuña Espinoza
et al. (2025): older inputs are processed at a coarser temporal resolution,
recent inputs are processed at a finer temporal resolution, and the resulting
packed sequence is passed through a single shared xLSTM backbone.

The higher-frequency output is the primary prediction. Lower-frequency output
is returned only for compatibility with NeuralHydrology's multi-frequency
output interface.

Reference
---------
Acuña Espinoza, E., Kratzert, F., Klotz, D., Gauch, M., Álvarez Chaves, M.,
Loritz, R., & Ehret, U. (2025). Technical note: An approach for handling
multiple temporal frequencies with different input dimensions using a single
LSTM cell. Hydrology and Earth System Sciences, 29(6), 1749-1758.
https://doi.org/10.5194/hess-29-1749-2025
"""

from __future__ import annotations

import os
import shutil
from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.datautils.utils import get_frequency_factor, sort_frequencies
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.utils.config import Config

# Configure CUDA-related paths required by the xLSTM sLSTM CUDA backend.
if "CUDA_HOME" not in os.environ:
    nvcc_path = shutil.which("nvcc")
    if nvcc_path is not None:
        os.environ["CUDA_HOME"] = os.path.dirname(os.path.dirname(nvcc_path))

cuda_home = os.environ.get("CUDA_HOME")
if cuda_home:
    cuda_home = os.path.expandvars(os.path.expanduser(cuda_home))

    if "CUDA_LIB" not in os.environ:
        for subdir in ("lib64", "lib", "lib/x86_64-linux-gnu"):
            lib_path = os.path.join(cuda_home, subdir)
            if os.path.exists(lib_path):
                os.environ["CUDA_LIB"] = lib_path
                break

    if "XLSTM_EXTRA_INCLUDE_PATHS" not in os.environ:
        include_path = os.path.join(cuda_home, "include")
        if os.path.exists(include_path):
            os.environ["XLSTM_EXTRA_INCLUDE_PATHS"] = include_path

from xlstm import (
    FeedForwardConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
)


class XLSTM(BaseModel):
    """Multi-frequency xLSTM model for NeuralHydrology."""

    module_parts = ["embedding_nets", "transition_layers", "xlstm", "dropout", "head"]

    def __init__(self, cfg: Config):
        super().__init__(cfg=cfg)

        # Match the active CUDA device to NeuralHydrology's configured device.
        if torch.cuda.is_available():
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            if world_size == 1:
                dev = torch.device(cfg.device)
                if dev.type == "cuda" and dev.index is not None:
                    torch.cuda.set_device(dev.index)
            print("[XLSTM init] cfg.device:", cfg.device, " current_device:", torch.cuda.current_device())

        if len(cfg.use_frequencies) < 2:
            raise ValueError("MF-xLSTM expects at least two frequencies, e.g. ['1D', '1h'].")

        # Frequencies are ordered from coarse to fine, e.g. ['1D', '1h'].
        self._frequencies = sort_frequencies(cfg.use_frequencies)

        self._seq_lengths = cfg.seq_length
        if not isinstance(self._seq_lengths, dict):
            raise ValueError("MF-xLSTM expects cfg.seq_length as a dictionary keyed by frequency.")

        # Determine how many coarse time steps overlap with the next finer input.
        # These overlapping steps are removed during replacement packing.
        self._slice_timestep: Dict[str, int] = {}
        for i, f in enumerate(self._frequencies[:-1]):
            f_next = self._frequencies[i + 1]
            factor = get_frequency_factor(f, f_next)
            if factor != int(factor):
                raise ValueError(f"Adjacent frequencies must be integer multiples: {f}->{f_next} factor={factor}.")
            factor = int(factor)

            L_next = int(self._seq_lengths[f_next])
            L_low = int(self._seq_lengths[f])
            if L_next % factor != 0:
                raise ValueError(f"seq_length[{f_next}]={L_next} must be divisible by factor={factor} for {f}->{f_next}.")

            slice_ts = L_next // factor
            if slice_ts <= 0 or slice_ts > L_low:
                raise ValueError(f"Invalid slice_ts={slice_ts} for {f}->{f_next} (L_low={L_low}, L_next={L_next}, factor={factor}).")

            self._slice_timestep[f] = slice_ts

        # Each frequency has its own NeuralHydrology input layer, followed by a
        # linear projection to the shared xLSTM hidden dimension.
        self.embedding_nets = nn.ModuleDict()
        self.transition_layers = nn.ModuleDict()

        for f in self._frequencies:
            freq_params = {
                "model": "mfxlstm",
                "dynamic_inputs": cfg.dynamic_inputs[f] if isinstance(cfg.dynamic_inputs, dict) else cfg.dynamic_inputs,
                "dynamics_embedding": cfg.dynamics_embedding,
                "static_attributes": cfg.static_attributes,
                "statics_embedding": cfg.statics_embedding,
                "use_basin_id_encoding": cfg.use_basin_id_encoding,
                "number_of_basins": cfg.number_of_basins,
                "head": cfg.head,
                "seq_length": cfg.seq_length[f],
            }
            fcfg = Config(freq_params)
            self.embedding_nets[f] = InputLayer(fcfg, embedding_type="full_model")
            self.transition_layers[f] = nn.Linear(self.embedding_nets[f].output_size, cfg.hidden_size)

        # xLSTM requires the full packed sequence length as context length.
        packed_len = self._calc_packed_len()

        xlstm_cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=cfg.xlstm_kernel_size,
                    qkv_proj_blocksize=4,
                    num_heads=cfg.xlstm_heads,
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda" if torch.cuda.is_available() else "vanilla",
                    num_heads=cfg.xlstm_heads,
                    conv1d_kernel_size=cfg.xlstm_kernel_size,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(
                    proj_factor=cfg.xlstm_proj_factor,
                    act_fn="gelu",
                ),
            ),
            context_length=packed_len,
            num_blocks=cfg.xlstm_num_blocks,
            embedding_dim=cfg.hidden_size,
            slstm_at=cfg.xlstm_slstm_at,
        )
        self.xlstm = xLSTMBlockStack(xlstm_cfg)

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        # The finest frequency is the main prediction frequency.
        self._target_freq = self._frequencies[-1]
        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

    def _calc_packed_len(self) -> int:
        """Return the sequence length after multi-frequency replacement packing."""
        L = int(self._seq_lengths[self._frequencies[0]])
        for i, f in enumerate(self._frequencies[:-1]):
            f_next = self._frequencies[i + 1]
            slice_ts = int(self._slice_timestep[f])
            L = (L - slice_ts) + int(self._seq_lengths[f_next])
        return int(L)

    def _embed_one_freq(self, data: dict, freq: str) -> torch.Tensor:
        """Embed one frequency-specific input sequence and project it to hidden size."""
        input_data = {
            "x_d": data[f"x_d_{freq}"],
            "x_s": data.get("x_s"),
            "x_one_hot": data.get("x_one_hot"),
        }
        if f"x_s_{freq}" in data:
            input_data["x_s"] = data[f"x_s_{freq}"]
        input_data = {k: v for k, v in input_data.items() if v is not None}

        x = self.embedding_nets[freq](input_data, concatenate_output=True)  # [S, B, E]
        x = x.transpose(0, 1).contiguous()                                  # [B, S, E]
        x = self.transition_layers[freq](x)                                  # [B, S, H]

        return x

    def _pack_multifreq(self, xs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Pack frequency-specific sequences by replacing overlapping coarse steps."""
        packed = xs[self._frequencies[0]]
        for i, f in enumerate(self._frequencies[:-1]):
            f_next = self._frequencies[i + 1]
            slice_ts = int(self._slice_timestep[f])
            packed = torch.cat([packed[:, :-slice_ts, :], xs[f_next]], dim=1)
        return packed

    def forward(self, data: dict) -> Dict[str, torch.Tensor]:
        """Run the MF-xLSTM forward pass."""
        # 1. Embed each frequency and build the packed multi-frequency sequence.
        xs = {f: self._embed_one_freq(data, f) for f in self._frequencies}
        x_packed = self._pack_multifreq(xs)       # [B, L, H]

        # 2. Process the packed sequence with a single shared xLSTM backbone.
        xlstm_out = self.xlstm(x_packed)          # [B, L, H]

        # 3. Use the finest-frequency hidden states as the primary prediction states.
        L_hi = int(self._seq_lengths[self._target_freq])
        h_hi = xlstm_out[:, -L_hi:, :]            # [B, L_hi, H]

        pred: Dict[str, torch.Tensor] = {}

        def _head_with_suffix(h: torch.Tensor, freq: str) -> Dict[str, torch.Tensor]:
            """Apply the output head and append the frequency suffix to output keys."""
            out = self.head(self.dropout(h))
            return {f"{k}_{freq}": v for k, v in out.items()}

        # Primary output at the finest frequency.
        pred.update(_head_with_suffix(h_hi, self._target_freq))

        # Compatibility outputs for coarser frequencies. These are derived by
        # average-pooling the finest-frequency hidden states, not by separate
        # frequency-specific recurrent branches.
        pln = getattr(self.cfg, "predict_last_n", None)
        if not isinstance(pln, dict):
            raise ValueError("MF-xLSTM expects cfg.predict_last_n as a dictionary keyed by frequency.")

        for f in self._frequencies[:-1]:
            factor = get_frequency_factor(f, self._target_freq)
            if factor != int(factor):
                raise ValueError(f"Cannot aggregate {self._target_freq} -> {f}: non-integer factor={factor}.")
            factor = int(factor)

            n_low = int(pln.get(f, 0))
            if n_low <= 0:
                continue

            need_hi = factor * n_low
            if need_hi > h_hi.shape[1]:
                raise ValueError(
                    f"predict_last_n[{f}]={n_low} requires {need_hi} steps of {self._target_freq}, "
                    f"but only have L_hi={h_hi.shape[1]}. Increase seq_length[{self._target_freq}] "
                    f"or reduce predict_last_n[{f}]."
                )

            h_tail = h_hi[:, -need_hi:, :]        # [B, need_hi, H]
            h_low = h_tail.reshape(h_tail.shape[0], n_low, factor, h_tail.shape[2]).mean(dim=2)

            pred.update(_head_with_suffix(h_low, f))

        # Plain y_hat is kept as a backwards-compatible alias to the finest-frequency output.
        pred["y_hat"] = pred[f"y_hat_{self._target_freq}"]

        return pred
