"""Small, hardware-friendly student network.

No recurrence (LSTM/GRU) anywhere — the backbone is either a Transformer encoder or a
dilated 1-D CNN, both of which parallelize well on CPU/GPU and export cleanly to ONNX.
Three heads share one encoder: MS2 fragment intensities, retention time, and CCS.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from ..chem import ION_TYPES
from ..data.encode import N_TOKENS, PAD_IDX
from ..data.encode import Batch


@dataclass(slots=True)
class StudentConfig:
    backbone: str = "transformer"  # "transformer" | "cnn"
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4  # transformer only
    kernel_size: int = 5  # cnn only
    dropout: float = 0.1
    max_len: int = 64
    max_charge: int = 8
    n_ion: int = len(ION_TYPES)
    # FFN expansion inside each transformer block. 4x is the classic default; 2x roughly
    # halves FFN FLOPs so the budget can go to depth (Rust/candle-friendly).
    ff_mult: int = 4
    # Activation everywhere (backbone + heads). "gelu" | "relu". gelu is the conventional
    # transformer default and what the arch sweeps used; relu is cheaper to fuse and
    # quantize, so revisit it when the candle/int8 runtime lands.
    activation: str = "gelu"
    # Where charge enters. False (default): charge leaves the trunk entirely — RT reads a
    # charge-free pooled rep so RT is *structurally* charge-invariant (a peptide's RT is a
    # per-peptide property; same stripped peptide at multiple charges must share one RT for
    # a self-consistent library — inductive bias, not something we hope training learns).
    # CCS concatenates charge_emb at its head, MS2 adds charge_emb per fragment site. True:
    # charge_emb added to every trunk position (all heads see it). A/B showed equal MAE;
    # False wins on library self-consistency (in_trunk wobbles RT ~0.02min across charges).
    charge_in_trunk: bool = False
    # Acquisition-context conditioning. A per-source context VECTOR (not id) enters at the
    # heads as a zero-init additive bias: ctx_acq drives MS2 + CCS (instrument / collision
    # energy / ion mobility), ctx_lc drives RT (chromatography). Zero-init => an untrained
    # or absent source reproduces the base model exactly. Fit only these 16-d vectors to
    # adapt to a new instrument/gradient (backbone frozen); swap a vector to swap setups.
    # 0 disables the context projections entirely.
    context_dim: int = 16

    def to_dict(self) -> dict:
        return asdict(self)

    def act_module(self) -> nn.Module:
        return {"gelu": nn.GELU, "relu": nn.ReLU}[self.activation]()


class _CNNBackbone(nn.Module):
    def __init__(self, cfg: StudentConfig) -> None:
        super().__init__()
        layers = []
        for i in range(cfg.n_layers):
            layers.append(
                nn.Conv1d(
                    cfg.d_model,
                    cfg.d_model,
                    cfg.kernel_size,
                    padding=(cfg.kernel_size // 2) * (2**i),
                    dilation=2**i,
                )
            )
            layers.append(cfg.act_module())
            layers.append(nn.Dropout(cfg.dropout))
        self.net = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
        # Zero padded columns at input and after every conv so a real position's
        # receptive field sees consistent zeros regardless of how much padding the batch
        # happens to carry (otherwise conv leaks pad -> batch-composition-dependent output).
        keep = None if pad_mask is None else (~pad_mask).unsqueeze(1).to(x.dtype)  # (B,1,L)
        h = x.transpose(1, 2)  # (B, d, L)
        if keep is not None:
            h = h * keep
        i = 0
        while i < len(self.net):
            conv, act, drop = self.net[i], self.net[i + 1], self.net[i + 2]
            y = drop(act(conv(h)))
            if keep is not None:
                y = y * keep
            h = y + h  # residual
            i += 3
        return h.transpose(1, 2)  # (B, L, d)


class _TransformerBackbone(nn.Module):
    def __init__(self, cfg: StudentConfig) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * cfg.ff_mult,
            dropout=cfg.dropout,
            batch_first=True,
            activation=cfg.activation,
        )
        self.net = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        return self.net(x, src_key_padding_mask=pad_mask)


class StudentModel(nn.Module):
    def __init__(self, cfg: StudentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.token_emb = nn.Embedding(N_TOKENS, d, padding_idx=PAD_IDX)
        self.pos_emb = nn.Embedding(cfg.max_len, d)
        self.charge_emb = nn.Embedding(cfg.max_charge + 1, d)
        self.mod_proj = nn.Linear(1, d)

        if cfg.backbone == "transformer":
            self.backbone: nn.Module = _TransformerBackbone(cfg)
        elif cfg.backbone == "cnn":
            self.backbone = _CNNBackbone(cfg)
        else:
            raise ValueError(f"unknown backbone {cfg.backbone!r}")

        # CCS head sees charge either from the trunk (in_dim=d) or concatenated at the head
        # (in_dim=2d) when charge is factored out. MS2/RT always take d (charge, when
        # factored out, is added — not concatenated — to the MS2 fragment features).
        ccs_in = d if cfg.charge_in_trunk else 2 * d
        self.ms2_head = nn.Sequential(nn.Linear(d, d), cfg.act_module(), nn.Linear(d, cfg.n_ion))
        self.rt_head = nn.Sequential(nn.Linear(d, d), cfg.act_module(), nn.Linear(d, 1))
        self.ccs_head = nn.Sequential(nn.Linear(ccs_in, d), cfg.act_module(), nn.Linear(d, 1))

        # Context projections: 16-d source vector -> per-head additive bias (d). Shared
        # ctx_acq feeds MS2 and CCS through separate projections; ctx_lc feeds RT. Only the
        # BIAS is zero-init (weights keep their random init): a zero context vector then maps
        # to a zero bias == base model, while the nonzero weights still pass gradient to the
        # context embeddings (zeroing weights too would strand the whole subsystem at 0).
        self.acq_to_ms2 = self.acq_to_ccs = self.lc_to_rt = None
        if cfg.context_dim:
            self.acq_to_ms2 = nn.Linear(cfg.context_dim, d)
            self.acq_to_ccs = nn.Linear(cfg.context_dim, d)
            self.lc_to_rt = nn.Linear(cfg.context_dim, d)
            for lin in (self.acq_to_ms2, self.acq_to_ccs, self.lc_to_rt):
                nn.init.zeros_(lin.bias)

        # Target normalization stats (set by the trainer, saved in the checkpoint).
        # Heads regress standardized targets; these map back to native units.
        self.register_buffer("rt_mean", torch.zeros(1))
        self.register_buffer("rt_std", torch.ones(1))
        self.register_buffer("ccs_mean", torch.zeros(1))
        self.register_buffer("ccs_std", torch.ones(1))

    def set_norm(self, rt_mean: float, rt_std: float, ccs_mean: float, ccs_std: float) -> None:
        self.rt_mean.fill_(rt_mean)
        self.rt_std.fill_(max(rt_std, 1e-6))
        self.ccs_mean.fill_(ccs_mean)
        self.ccs_std.fill_(max(ccs_std, 1e-6))

    def denormalize(self, out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Map standardized rt/ccs predictions back to native units (in-place-safe copy)."""
        return {
            "ms2": out["ms2"],
            "rt": out["rt"] * self.rt_std + self.rt_mean,
            "ccs": out["ccs"] * self.ccs_std + self.ccs_mean,
        }

    def _embed_tensors(
        self, tokens: torch.Tensor, mod_delta: torch.Tensor, charge: torch.Tensor
    ) -> torch.Tensor:
        length = tokens.shape[1]
        pos = torch.arange(length, device=tokens.device).unsqueeze(0)
        x = self.token_emb(tokens) + self.pos_emb(pos)
        x = x + self.mod_proj(mod_delta.unsqueeze(-1))
        if self.cfg.charge_in_trunk:
            x = x + self.charge_emb(charge).unsqueeze(1)
        return x

    def _apply_heads(
        self,
        h: torch.Tensor,
        pooled: torch.Tensor,
        charge: torch.Tensor,
        ctx_acq: torch.Tensor | None = None,
        ctx_lc: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the three heads on backbone output. Returns standardized (ms2, rt, ccs).

        When charge is factored out of the trunk it re-enters here: added per fragment
        site for MS2, concatenated for CCS, and never for RT (charge-invariant). Optional
        acquisition context enters the same way (zero-init additive bias): ctx_acq shifts
        the MS2 and CCS head inputs, ctx_lc shifts the RT head input.
        """
        frag_feat = 0.5 * (h[:, :-1] + h[:, 1:])  # (B, L-1, d)
        ms2_feat, ccs_feat, rt_feat = frag_feat, pooled, pooled
        if ctx_acq is not None:
            ms2_feat = ms2_feat + self.acq_to_ms2(ctx_acq).unsqueeze(1)
            ccs_feat = ccs_feat + self.acq_to_ccs(ctx_acq)
        if ctx_lc is not None:
            rt_feat = rt_feat + self.lc_to_rt(ctx_lc)

        if self.cfg.charge_in_trunk:
            ms2 = torch.sigmoid(self.ms2_head(ms2_feat))
            ccs = self.ccs_head(ccs_feat).squeeze(-1)
        else:
            ce = self.charge_emb(charge)  # (B, d)
            ms2 = torch.sigmoid(self.ms2_head(ms2_feat + ce.unsqueeze(1)))
            ccs = self.ccs_head(torch.cat([ccs_feat, ce], dim=-1)).squeeze(-1)
        rt = self.rt_head(rt_feat).squeeze(-1)
        return ms2, rt, ccs

    def _embed(self, batch: Batch) -> torch.Tensor:
        return self._embed_tensors(batch.tokens, batch.mod_delta, batch.charge)

    def forward_dense(
        self,
        tokens: torch.Tensor,
        mod_delta: torch.Tensor,
        charge: torch.Tensor,
        ctx_acq: torch.Tensor | None = None,
        ctx_lc: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mask-free forward for same-length batches; returns denormalized (ms2, rt, ccs).

        This is the inference/ONNX path: no padding, so no masks — attention and pooling
        run dense. Returns plain tensors (not a dict) so it exports cleanly to ONNX. Context
        vectors are optional; bake them as constants for a fixed-instrument export.
        """
        x = self._embed_tensors(tokens, mod_delta, charge)
        # Dense/bucketed inputs have no padding, so no mask. Passing None (vs an all-False
        # mask) also avoids TransformerEncoder's eval fast-path NestedTensor packing, whose
        # aten::_nested_tensor_from_mask_left_aligned op is unimplemented on MPS.
        h = self.backbone(x, None)
        pooled = h.mean(dim=1)
        ms2, rt, ccs = self._apply_heads(h, pooled, charge, ctx_acq, ctx_lc)
        rt = rt * self.rt_std + self.rt_mean
        ccs = ccs * self.ccs_std + self.ccs_mean
        return ms2, rt, ccs

    @staticmethod
    def _masked_mean(h: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        keep = (~pad_mask).float().unsqueeze(-1)  # (B, L, 1)
        return (h * keep).sum(1) / keep.sum(1).clamp_min(1.0)

    def forward(
        self,
        batch: Batch,
        ctx_acq: torch.Tensor | None = None,
        ctx_lc: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x = self._embed(batch)
        h = self.backbone(x, batch.pad_mask)  # (B, L, d)
        pooled = self._masked_mean(h, batch.pad_mask)  # (B, d)
        ms2, rt, ccs = self._apply_heads(h, pooled, batch.charge, ctx_acq, ctx_lc)
        return {"ms2": ms2, "rt": rt, "ccs": ccs}

    def forward_context(
        self,
        batch: Batch,
        ctx_acq: torch.Tensor | None = None,
        ctx_lc: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """One backbone pass giving BOTH the context-conditioned heads and the context-free
        base RT. Used by the real-speclib regime: ``rt`` (ctx_lc-conditioned) is supervised
        against raw retention time, while ``rt_base`` (no ctx_lc) is supervised against the
        run-independent iRT — so ctx_lc learns only each run's LC deviation.
        """
        x = self._embed(batch)
        h = self.backbone(x, batch.pad_mask)
        pooled = self._masked_mean(h, batch.pad_mask)
        ms2, rt, ccs = self._apply_heads(h, pooled, batch.charge, ctx_acq, ctx_lc)
        rt_base = self.rt_head(pooled).squeeze(-1)  # context-free (iRT frame)
        return {"ms2": ms2, "rt": rt, "ccs": ccs, "rt_base": rt_base}

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
