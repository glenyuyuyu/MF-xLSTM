"""
Multi-frequency CudaLSTM for NeuralHydrology.

Design:
- Keeps the multi-frequency replacement packing logic from x_lstm.py.
- Replaces the xLSTM backbone with PyTorch's CUDA-backed nn.LSTM.
- Removes the learnable frequency embedding mechanism entirely.
- Predicts only the highest frequency (last entry in cfg.use_frequencies),
  matching the original simplified x_lstm.py behavior.

Expected NH batch keys:
- x_d_{freq}
- optional x_s or x_s_{freq}
- optional x_one_hot
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.datautils.utils import get_frequency_factor, sort_frequencies
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.utils.config import Config


class MTSLSTM(BaseModel):
    """Multi-frequency LSTM with replacement packing.

    Notes
    -----
    This class preserves the overall data flow of the original multi-frequency x_lstm.py:
        per-frequency InputLayer -> replacement packing -> single backbone -> head

    The only architectural change is that the xLSTM backbone is replaced by a standard
    PyTorch nn.LSTM (CUDA-accelerated when run on GPU). The learnable frequency embedding
    used in the original x_lstm.py is removed.

    The model predicts only the highest-frequency target, i.e. the last frequency in
    cfg.use_frequencies after sorting.
    """

    module_parts = ["embedding_nets", "transition_layers", "lstm", "dropout", "head"]

    def __init__(self, cfg: Config):
        super().__init__(cfg=cfg)

        if len(cfg.use_frequencies) < 2:
            raise ValueError("This model expects at least two frequencies, e.g. ['1D', '1h'].")

        self._frequencies = sort_frequencies(cfg.use_frequencies)  # low -> high

        self._seq_lengths = cfg.seq_length
        if not isinstance(self._seq_lengths, dict):
            raise ValueError("This model expects cfg.seq_length to be a dict keyed by frequency.")

        # ------------------------------------------------------------------
        # Replacement packing setup: lower-frequency overlapping timesteps are
        # replaced by higher-frequency windows instead of duplicated.
        # ------------------------------------------------------------------
        self._slice_timestep: Dict[str, int] = {}
        for i, f in enumerate(self._frequencies[:-1]):
            f_next = self._frequencies[i + 1]
            factor = get_frequency_factor(f, f_next)

            if factor != int(factor):
                raise ValueError(
                    f"Adjacent frequencies must be integer multiples: {f} -> {f_next}, factor={factor}."
                )
            factor = int(factor)

            l_next = int(self._seq_lengths[f_next])
            l_low = int(self._seq_lengths[f])

            if l_next % factor != 0:
                raise ValueError(
                    f"seq_length[{f_next}]={l_next} must be divisible by factor={factor} for {f} -> {f_next}."
                )

            slice_ts = l_next // factor
            if slice_ts <= 0 or slice_ts > l_low:
                raise ValueError(
                    f"Invalid slice_ts={slice_ts} for {f} -> {f_next} "
                    f"(L_low={l_low}, L_next={l_next}, factor={factor})."
                )

            self._slice_timestep[f] = slice_ts

        # ------------------------------------------------------------------
        # Per-frequency input layers. We keep the original x_lstm.py pattern:
        # each frequency gets its own InputLayer, then projected to hidden_size
        # so all frequencies share one common backbone input dimension.
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Single recurrent backbone: replace xLSTMBlockStack with nn.LSTM.
        # We use batch_first=True because the packed sequence is assembled as
        # [batch, seq, hidden].
        # ------------------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=cfg.hidden_size,
            hidden_size=cfg.hidden_size,
            batch_first=True,
        )

        self.dropout = nn.Dropout(p=cfg.output_dropout)
        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

        self._target_freq = self._frequencies[-1]

        self._reset_parameters()

    def _reset_parameters(self):
        """Special initialization of forget gate bias, matching CudaLSTM behavior."""
        if self.cfg.initial_forget_bias is not None:
            hidden_size = self.cfg.hidden_size
            self.lstm.bias_hh_l0.data[hidden_size:2 * hidden_size] = self.cfg.initial_forget_bias

    def _embed_one_freq(self, data: dict, freq: str) -> torch.Tensor:
        """Embed one frequency stream and project to the common hidden dimension.

        Returns
        -------
        torch.Tensor
            Tensor of shape [batch, seq, hidden_size].
        """
        input_data = {
            "x_d": data[f"x_d_{freq}"],
            "x_s": data.get("x_s"),
            "x_one_hot": data.get("x_one_hot"),
        }

        if f"x_s_{freq}" in data:
            input_data["x_s"] = data[f"x_s_{freq}"]

        input_data = {k: v for k, v in input_data.items() if v is not None}

        x = self.embedding_nets[freq](input_data, concatenate_output=True)  # [seq, batch, emb]
        x = x.transpose(0, 1).contiguous()  # [batch, seq, emb]
        x = self.transition_layers[freq](x)  # [batch, seq, hidden_size]
        return x

    def _pack_multifreq(self, xs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Replacement-pack multiple frequencies into one sequence.

        Example
        -------
        If the low-frequency tail overlaps in time with the higher-frequency window,
        the overlapping low-frequency timesteps are removed and replaced by the
        higher-frequency sequence.
        """
        packed = xs[self._frequencies[0]]

        for i, f in enumerate(self._frequencies[:-1]):
            f_next = self._frequencies[i + 1]
            slice_ts = int(self._slice_timestep[f])
            packed = torch.cat([packed[:, :-slice_ts, :], xs[f_next]], dim=1)

        return packed

    def forward(self, data: dict) -> Dict[str, torch.Tensor]:
        """Run the multi-frequency packed sequence through the LSTM backbone.

        Outputs
        -------
        Dict[str, torch.Tensor]
            - always returns highest-frequency prediction with suffix, e.g. y_hat_1H
            - additionally returns lower-frequency predictions by aggregating the
            highest-frequency hidden states into lower-frequency steps
            - keeps plain 'y_hat' as backward-compatible alias to the target frequency
        """
        # ---- backbone on packed multi-frequency sequence ----
        xs = {f: self._embed_one_freq(data, f) for f in self._frequencies}
        x_packed = self._pack_multifreq(xs)  # [B, L, H]

        lstm_output, (h_n, c_n) = self.lstm(x_packed)  # [B, L, H]

        # ---- highest-frequency hidden states ----
        l_hi = int(self._seq_lengths[self._target_freq])
        h_hi = lstm_output[:, -l_hi:, :]  # [B, L_hi, H]

        pred: Dict[str, torch.Tensor] = {}

        def _head_with_suffix(h: torch.Tensor, freq: str) -> Dict[str, torch.Tensor]:
            out = self.head(self.dropout(h))
            out = {f"{k}_{freq}": v for k, v in out.items()}
            return out

        # 1) highest-frequency prediction
        pred.update(_head_with_suffix(h_hi, self._target_freq))

        # 2) lower-frequency predictions via aggregation of highest-frequency hidden states
        pln = getattr(self.cfg, "predict_last_n", None)
        if not isinstance(pln, dict):
            raise ValueError(
                "This multi-frequency LSTM expects cfg.predict_last_n as a dict keyed by frequency."
            )

        for f in self._frequencies[:-1]:
            factor = get_frequency_factor(f, self._target_freq)
            if factor != int(factor):
                raise ValueError(
                    f"Cannot aggregate {self._target_freq} -> {f}: non-integer factor={factor}."
                )
            factor = int(factor)

            n_low = int(pln.get(f, 0))
            if n_low <= 0:
                continue

            need_hi = factor * n_low
            if need_hi > h_hi.shape[1]:
                raise ValueError(
                    f"predict_last_n[{f}]={n_low} requires {need_hi} steps of {self._target_freq}, "
                    f"but only have L_hi={h_hi.shape[1]}. "
                    f"Increase seq_length[{self._target_freq}] or reduce predict_last_n[{f}]."
                )

            # take the last need_hi high-frequency steps and pool them into n_low low-frequency steps
            h_tail = h_hi[:, -need_hi:, :]  # [B, need_hi, H]
            h_low = h_tail.reshape(
                h_tail.shape[0], n_low, factor, h_tail.shape[2]
            ).mean(dim=2)  # [B, n_low, H]

            pred.update(_head_with_suffix(h_low, f))

        # 3) backward-compatible alias
        pred["y_hat"] = pred[f"y_hat_{self._target_freq}"]

        # 4) optional intermediate states for debugging/inspection
        pred["lstm_output"] = lstm_output
        pred["h_n"] = h_n.transpose(0, 1)  # [batch, num_layers, hidden_size]
        pred["c_n"] = c_n.transpose(0, 1)  # [batch, num_layers, hidden_size]

        return pred
