"""Small, hardware-friendly student network.

No recurrence (LSTM/GRU) anywhere — the backbone is either a Transformer encoder or a
dilated 1-D CNN, both of which parallelize well on CPU/GPU and export cleanly to ONNX.
Three heads share one encoder: MS2 fragment intensities, retention time, and CCS.
"""

from __future__ import annotations

import math
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
    # Activation everywhere (backbone + heads). "gelu" | "relu" | "leaky_relu". gelu is the
    # conventional transformer default; the ReLU variants are explicit retraining experiments,
    # never silent inference substitutions.
    activation: str = "gelu"
    # Acquisition-context conditioning. A per-source context VECTOR (not id) enters at the
    # heads as a zero-init additive bias: ms_context drives MS2 fragments (instrument /
    # collision energy / fragmentation), chrom_context drives RT (chromatography). CCS takes
    # no acquisition context (peptide + charge only). Zero-init => an untrained or absent
    # source reproduces the base model exactly. Fit only these 16-d vectors to adapt to a
    # new instrument/gradient (backbone frozen); swap a vector to swap setups. 0 disables the
    # context projections entirely.
    context_dim: int = 16

    # Modification encoding. A mod enters the model through EITHER the compositional encoder
    # (element counts over C,H,N,O,S,P) or the mass-only encoder — never both. They share one
    # embedding space, held together by the mod_align loss. n_elements is a frozen input
    # contract; changing it invalidates every checkpoint.
    n_elements: int = 6
    mass_fourier_k: int = 16
    # Probability that a composition-bearing site is routed through the mass encoder during
    # training, so the backbone experiences the fallback's error instead of meeting it cold at
    # inference. Eval always routes to the compositional encoder.
    mass_swap_p: float = 0.15

    def to_dict(self) -> dict:
        return asdict(self)

    def act_module(self) -> nn.Module:
        if self.activation == "gelu":
            return nn.GELU()
        if self.activation == "relu":
            return nn.ReLU()
        if self.activation == "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.01)
        raise ValueError(f"unknown activation {self.activation!r}")


class FourierFeatures(nn.Module):
    """Sinusoidal expansion of a scalar over a geometric wavelength ladder.

    Wavelengths run from WAVELENGTH_MAX down to WAVELENGTH_MIN Daltons, spanning the range
    modification deltas actually occupy. This exists because a Linear(1, d) cannot work: it
    maps every modification onto a single line through its bias, while the compositional
    encoder's outputs are not collinear in mass, so the alignment target would be unreachable.
    """

    WAVELENGTH_MAX = 5000.0
    WAVELENGTH_MIN = 0.1

    def __init__(self, k: int) -> None:
        super().__init__()
        lam = torch.logspace(
            math.log10(self.WAVELENGTH_MAX), math.log10(self.WAVELENGTH_MIN), k
        )
        # Buffer, not a parameter: the ladder is fixed, and it must travel with the module
        # into checkpoints and ONNX exports.
        self.register_buffer("freq", 2.0 * math.pi / lam)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x.unsqueeze(-1) * self.freq  # (..., k)
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)  # (..., 2k)


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
            # Pass a module so alternatives such as Leaky ReLU are supported; PyTorch's
            # string shorthand only recognizes "relu" and "gelu".
            activation=cfg.act_module(),
        )
        # enable_nested_tensor=False: the eval fast path packs padded batches via
        # aten::_nested_tensor_from_mask_left_aligned, which is unimplemented on MPS — it
        # crashes the moment a masked batch is run in eval mode (i.e. every validation step
        # of the real-speclib regime). The packing is a throughput optimization only, so
        # disabling it is numerically identical. It was already off for presets with an odd
        # head count, which torch refuses to pack; this makes every preset take one path.
        self.net = nn.TransformerEncoder(layer, num_layers=cfg.n_layers, enable_nested_tensor=False)

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
        # Compositional encoder. Linear on purpose: it makes the space compositional, so the
        # composition of two mods embeds as the sum of their vectors.
        self.comp_enc = nn.Linear(cfg.n_elements, d)
        # Mass-only fallback. Nonlinear over a Fourier expansion (see FourierFeatures).
        self.mass_enc = nn.Sequential(
            FourierFeatures(cfg.mass_fourier_k),
            nn.Linear(2 * cfg.mass_fourier_k, d),
            cfg.act_module(),
            nn.Linear(d, d),
        )
        nn.init.zeros_(self.comp_enc.bias)
        nn.init.zeros_(self.mass_enc[-1].bias)

        if cfg.backbone == "transformer":
            self.backbone: nn.Module = _TransformerBackbone(cfg)
        elif cfg.backbone == "cnn":
            self.backbone = _CNNBackbone(cfg)
        else:
            raise ValueError(f"unknown backbone {cfg.backbone!r}")

        # Charge is factored out of the trunk (RT must stay charge-invariant — the RT label is
        # id-time, so its charge-dependence is measurement artifact, not signal), so it re-enters
        # only at the heads: concatenated to CCS (in_dim=2d) and added per fragment site to MS2.
        ccs_in = 2 * d
        self.ms2_head = nn.Sequential(nn.Linear(d, d), cfg.act_module(), nn.Linear(d, cfg.n_ion))
        self.rt_head = nn.Sequential(nn.Linear(d, d), cfg.act_module(), nn.Linear(d, 1))
        self.ccs_head = nn.Sequential(nn.Linear(ccs_in, d), cfg.act_module(), nn.Linear(d, 1))

        # Context projections (zero-init bias -> zero context = base). ms_context feeds the
        # per-fragment features; chrom_context feeds the RT head. CCS takes NO acquisition
        # context (peptide + charge only).
        self.ms_to_frag = self.chrom_to_rt = None
        if cfg.context_dim:
            self.ms_to_frag = nn.Linear(cfg.context_dim, d)
            self.chrom_to_rt = nn.Linear(cfg.context_dim, d)
            for lin in (self.ms_to_frag, self.chrom_to_rt):
                nn.init.zeros_(lin.bias)

        # Target normalization stats (set by the trainer, saved in the checkpoint).
        # Heads regress standardized targets; these map back to native units.
        self.register_buffer("rt_mean", torch.zeros(1))
        self.register_buffer("rt_std", torch.ones(1))
        self.register_buffer("ccs_mean", torch.zeros(1))
        self.register_buffer("ccs_std", torch.ones(1))
        # Whether the RT affine has been established. Explicit state, not inferred from
        # rt_mean==0/rt_std==1: those are legitimate values for a standardized frame, and
        # reading "unset" out of them is the sentinel-inference pattern this codebase
        # rejects elsewhere. The rule this exists to enforce: RT/CCS scale is ONE global
        # affine, set once at cold start, never re-established when a dataset is added —
        # per-dataset RT variation belongs to the ChromRunbook.
        self.register_buffer("norm_established", torch.zeros(1, dtype=torch.bool))

    def set_norm(
        self,
        rt_mean: float | None = None,
        rt_std: float | None = None,
        ccs_mean: float | None = None,
        ccs_std: float | None = None,
        *,
        force: bool = False,
    ) -> None:
        """Set target normalization; ``None`` leaves that statistic untouched.

        The RT affine is set ONCE, at cold start. A second attempt to establish it raises
        unless ``force=True``: re-standardizing mid-curriculum recalibrates an already-trained
        head against a new frame, and per-dataset RT variation is the ChromRunbook's job, not
        the norm's. Refusing loudly beats silently accepting the overwrite or silently
        ignoring the call — either one leaves the caller believing something that is not true.

        A regime with no data for a property MUST pass None rather than a placeholder.
        Writing (0.0, 1.0) over a calibration an earlier stage learned does not disable the
        head — it leaves a trained head whose outputs denormalize to raw standardized values,
        which look like plausible small numbers instead of native units. That is how a
        pretrain->train pipeline silently produced negative CCS.

        Non-finite values are rejected for the same reason: a NaN std would surface as a
        confident, meaningless prediction rather than a failure.
        """
        sets_rt = rt_mean is not None or rt_std is not None
        if sets_rt and bool(self.norm_established) and not force:
            raise ValueError(
                "set_norm: the RT affine is already established "
                f"(mean {float(self.rt_mean):.6g}, std {float(self.rt_std):.6g}); refusing to "
                f"re-establish it from (mean {rt_mean!r}, std {rt_std!r}). RT/CCS scale is one "
                "global affine fixed at cold start — per-dataset RT variation belongs to the "
                "ChromRunbook. Pass force=True only if you intend to discard the existing "
                "calibration of an already-trained head."
            )
        for name, value in (
            ("rt_mean", rt_mean), ("rt_std", rt_std),
            ("ccs_mean", ccs_mean), ("ccs_std", ccs_std),
        ):
            if value is None:
                continue
            if not math.isfinite(value):
                raise ValueError(f"set_norm: {name} must be finite, got {value!r}")
            buf = getattr(self, name)
            buf.fill_(max(value, 1e-6) if name.endswith("_std") else value)
        if rt_mean is not None or rt_std is not None:
            self.norm_established.fill_(True)

    def standardize_rt(self, rt: torch.Tensor) -> torch.Tensor:
        """Native-unit RT -> the head's standardized space (the target for training)."""
        return (rt - self.rt_mean) / self.rt_std

    def standardize_ccs(self, ccs: torch.Tensor) -> torch.Tensor:
        """Native-unit CCS -> the head's standardized space."""
        return (ccs - self.ccs_mean) / self.ccs_std

    def unstandardize_rt(self, rt: torch.Tensor) -> torch.Tensor:
        """Standardized RT prediction -> native units (inverse of :meth:`standardize_rt`)."""
        return rt * self.rt_std + self.rt_mean

    def denormalize(self, out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Map standardized rt/ccs predictions back to native units (in-place-safe copy)."""
        return {
            "ms2": out["ms2"],
            "rt": self.unstandardize_rt(out["rt"]),
            "ccs": out["ccs"] * self.ccs_std + self.ccs_mean,
        }

    def _mod_vectors(
        self,
        mod_comp: torch.Tensor,
        mod_mass: torch.Tensor,
        mod_present: torch.Tensor,
        mod_named: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (routed mod vector, comp-encoder output, mass-encoder output).

        Both encoders always run — the align loss needs both — but only one reaches the
        backbone per site. Unmodified positions are zeroed by mod_present, not by encoder
        behavior: the Fourier expansion of a zero mass is not a zero vector.
        """
        g = self.comp_enc(mod_comp)  # (B, T, d)
        m = self.mass_enc(mod_mass)  # (B, T, d)
        use_g = mod_named
        if self.training and self.cfg.mass_swap_p > 0.0:
            swap = torch.rand_like(mod_mass) < self.cfg.mass_swap_p
            use_g = use_g & ~swap
        vec = torch.where(use_g.unsqueeze(-1), g, m)
        return vec * mod_present.unsqueeze(-1).to(vec.dtype), g, m

    def _embed_tensors(
        self,
        tokens: torch.Tensor,
        mod_comp: torch.Tensor,
        mod_mass: torch.Tensor,
        mod_present: torch.Tensor,
        mod_named: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = tokens.shape[1]
        pos = torch.arange(length, device=tokens.device).unsqueeze(0)
        mod_vec, g, m = self._mod_vectors(mod_comp, mod_mass, mod_present, mod_named)
        return self.token_emb(tokens) + self.pos_emb(pos) + mod_vec, g, m

    def _apply_heads(
        self,
        h: torch.Tensor,
        pooled: torch.Tensor,
        charge: torch.Tensor,
        ms_context: torch.Tensor | None = None,
        chrom_context: torch.Tensor | None = None,
        chrom_affine: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the three heads. ms_context (broadcast) shifts the per-fragment features;
        chrom_context shifts the RT head; CCS is peptide + charge only. Charge re-enters here
        (it is factored out of the trunk): added per fragment site for MS2, concatenated for CCS.
        RT never sees charge — it stays structurally charge-invariant.

        chrom_affine is a per-dataset ``(scale, shift)`` applied to the RT head's OUTPUT, in
        standardized space. It exists because chrom_context is an additive bias in feature
        space and cannot express a rescale, while a dataset's raw RT differs from the iRT frame
        by scale as well as offset. None means identity, so every context-free path is
        bit-identical to a model without it.
        """
        frag_feat = 0.5 * (h[:, :-1] + h[:, 1:])  # (B, L-1, d)
        ms2_feat, ccs_feat, rt_feat = frag_feat, pooled, pooled
        if ms_context is not None:
            ms2_feat = ms2_feat + self.ms_to_frag(ms_context).unsqueeze(1)
        if chrom_context is not None:
            rt_feat = rt_feat + self.chrom_to_rt(chrom_context)

        ce = self.charge_emb(charge)  # (B, d)
        ms2 = torch.sigmoid(self.ms2_head(ms2_feat + ce.unsqueeze(1)))
        ccs = self.ccs_head(torch.cat([ccs_feat, ce], dim=-1)).squeeze(-1)
        rt = self.rt_head(rt_feat).squeeze(-1)
        if chrom_affine is not None:
            scale, shift = chrom_affine
            rt = scale * rt + shift
        return ms2, rt, ccs

    def _embed(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._embed_tensors(
            batch.tokens, batch.mod_comp, batch.mod_mass, batch.mod_present, batch.mod_named,
        )

    def forward_dense(
        self,
        tokens: torch.Tensor,
        mod_comp: torch.Tensor,
        mod_mass: torch.Tensor,
        mod_present: torch.Tensor,
        mod_named: torch.Tensor,
        charge: torch.Tensor,
        ms_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mask-free forward for same-length batches; returns denormalized (ms2, rt, ccs).

        This is the inference/ONNX path: no padding, so no masks — attention and pooling
        run dense. Returns plain tensors (not a dict) so it exports cleanly to ONNX. Takes
        MS context only (RT/CCS need no acquisition context here); bake it as a constant for
        a fixed-instrument export.
        """
        x, _, _ = self._embed_tensors(tokens, mod_comp, mod_mass, mod_present, mod_named)
        # Dense/bucketed inputs have no padding, so no mask. Passing None (vs an all-False
        # mask) also avoids TransformerEncoder's eval fast-path NestedTensor packing, whose
        # aten::_nested_tensor_from_mask_left_aligned op is unimplemented on MPS.
        h = self.backbone(x, None)
        pooled = h.mean(dim=1)
        ms2, rt, ccs = self._apply_heads(h, pooled, charge, ms_context, None)
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
        ms_context: torch.Tensor | None = None,
        chrom_context: torch.Tensor | None = None,
        chrom_affine: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        x, g, m = self._embed(batch)
        h = self.backbone(x, batch.pad_mask)  # (B, L, d)
        pooled = self._masked_mean(h, batch.pad_mask)  # (B, d)
        ms2, rt, ccs = self._apply_heads(
            h, pooled, batch.charge, ms_context, chrom_context, chrom_affine
        )
        return {"ms2": ms2, "rt": rt, "ccs": ccs, "mod_g": g, "mod_m": m}

    def forward_context(
        self,
        batch: Batch,
        ms_context: torch.Tensor | None = None,
        chrom_context: torch.Tensor | None = None,
        chrom_affine: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """One backbone pass giving the chrom-conditioned heads AND the context-free base RT
        (``rt_base``, no chrom_context = iRT frame), so the real regime supervises rt->raw RT
        and rt_base->iRT.

        ``rt_base`` deliberately sees neither chrom_context nor chrom_affine: it is the iRT
        anchor, and conditioning it on the dataset would destroy the frame it defines."""
        x, g, m = self._embed(batch)
        h = self.backbone(x, batch.pad_mask)
        pooled = self._masked_mean(h, batch.pad_mask)
        ms2, rt, ccs = self._apply_heads(
            h, pooled, batch.charge, ms_context, chrom_context, chrom_affine
        )
        rt_base = self.rt_head(pooled).squeeze(-1)
        return {"ms2": ms2, "rt": rt, "ccs": ccs, "rt_base": rt_base, "mod_g": g, "mod_m": m}

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
