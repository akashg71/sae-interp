"""The sparse autoencoder (classic L1 ReLU SAE).

This is the model we *implement ourselves* (the whole point of the project is to
show we understand SAEs, not to hide them behind a library). Phases 1-2 use a
pretrained SAE from SAELens; Phase 3 trains *this* one.

Architecture (for an input activation ``x ∈ R^d``, d = d_model):

    f      = ReLU( (x - b_dec) @ W_enc.T + b_enc )      # sparse feature codes, f ∈ R^{d_sae}
    x_hat  = f @ W_dec.T + b_dec                         # reconstruction

    L      = ||x - x_hat||^2  +  λ * ||f||_1

Key design choices and *why* (these are the non-obvious bits):

1. **Subtract ``b_dec`` before encoding.** ``b_dec`` acts as a learned "center" of
   the data. Centering before the encoder means the dictionary directions only have
   to explain *deviations* from the mean activation, which empirically helps.

2. **Unit-norm decoder columns.** Each column of ``W_dec`` is a dictionary direction
   ("feature direction"). We constrain them to unit L2 norm so that the feature
   magnitude lives entirely in ``f`` (not split ambiguously between f and the column
   norm). Without this, the L1 penalty could be gamed by shrinking f and growing the
   column norm. We enforce it two ways (use both): (a) project the *gradient* to be
   orthogonal to each column before the optimiser step, then (b) renormalise columns
   to unit norm after the step. (a) keeps Adam's statistics sane; (b) corrects drift.

3. **L1 on f drives sparsity.** λ (``l1_coeff``) controls where we sit on the
   sparsity↔reconstruction frontier. Higher λ -> sparser (lower L0), worse recon.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEOutput:
    """Bundle of everything a forward pass produces, so callers don't juggle tuples."""

    x_hat: torch.Tensor       # reconstruction, shape (..., d_in)
    feats: torch.Tensor       # feature activations f, shape (..., d_sae)
    recon_loss: torch.Tensor  # scalar mean reconstruction MSE (summed over d, mean over batch)
    l1_loss: torch.Tensor     # scalar λ * mean L1 of f
    loss: torch.Tensor        # scalar total loss = recon_loss + l1_loss
    l0: torch.Tensor          # scalar mean number of active features per token (for logging)


class SparseAutoencoder(nn.Module):
    """Classic L1 ReLU sparse autoencoder.

    Parameters
    ----------
    d_in:
        Input/output dimensionality (= model d_model, e.g. 768 for GPT-2-small).
    d_sae:
        Dictionary size (number of features), typically ``expansion_factor * d_in``.
    l1_coeff:
        Sparsity penalty weight λ.
    dtype:
        Parameter dtype. Keep float32 for training stability even on MPS.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        l1_coeff: float,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.l1_coeff = l1_coeff

        # Decoder bias doubles as the data "center" subtracted before encoding.
        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype))
        # Encoder: maps centered activations -> features. Shape (d_sae, d_in).
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_in, dtype=dtype))
        self.b_enc = nn.Parameter(torch.zeros(d_sae, dtype=dtype))
        # Decoder: maps features -> reconstruction. Shape (d_in, d_sae); columns are
        # the dictionary directions, constrained to unit norm.
        self.W_dec = nn.Parameter(torch.empty(d_in, d_sae, dtype=dtype))

        self._init_parameters()

    # ------------------------------------------------------------------ init
    def _init_parameters(self) -> None:
        # Kaiming init for the encoder is fine. For the decoder, a common and robust
        # choice is to initialise it as the transpose of the encoder, then unit-norm
        # the columns — this gives the autoencoder a sensible starting point where
        # encode/decode are roughly inverse.
        nn.init.kaiming_uniform_(self.W_enc)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t())
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Renormalise each decoder column (dictionary direction) to unit L2 norm."""
        norms = self.W_dec.data.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.W_dec.data /= norms

    @torch.no_grad()
    def remove_parallel_decoder_grad(self) -> None:
        """Project out the component of ``W_dec.grad`` parallel to each column.

        Call this *after* ``loss.backward()`` and *before* ``optimizer.step()``.
        Rationale: we want to keep columns unit-norm. The radial (parallel) part of
        the gradient is exactly the part that would change a column's length; removing
        it means the optimiser only rotates directions, so Adam's momentum/variance
        aren't polluted by length changes we're about to undo via renormalisation.
        """
        if self.W_dec.grad is None:
            return
        W = self.W_dec.data                      # (d_in, d_sae)
        g = self.W_dec.grad                       # (d_in, d_sae)
        # For each column w (unit norm), remove (g·w) w.
        parallel = (g * W).sum(dim=0, keepdim=True) * W
        self.W_dec.grad -= parallel

    # ----------------------------------------------------------------- encode
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Activations -> sparse feature codes f = ReLU((x - b_dec) W_enc^T + b_enc)."""
        return F.relu((x - self.b_dec) @ self.W_enc.t() + self.b_enc)

    def decode(self, feats: torch.Tensor) -> torch.Tensor:
        """Feature codes -> reconstruction x_hat = f W_dec^T + b_dec."""
        return feats @ self.W_dec.t() + self.b_dec

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> SAEOutput:
        feats = self.encode(x)
        x_hat = self.decode(feats)

        # Reconstruction loss: MSE summed over the feature dim, averaged over the
        # batch. Summing over d (rather than mean) is the SAE-literature convention
        # and keeps the loss scale comparable to the L1 term.
        recon_loss = (x_hat - x).pow(2).sum(-1).mean()
        l1_loss = self.l1_coeff * feats.abs().sum(-1).mean()
        loss = recon_loss + l1_loss
        l0 = (feats > 0).float().sum(-1).mean()

        return SAEOutput(
            x_hat=x_hat,
            feats=feats,
            recon_loss=recon_loss,
            l1_loss=l1_loss,
            loss=loss,
            l0=l0,
        )

    # ----------------------------------------------------------- persistence
    @torch.no_grad()
    def set_decoder_bias(self, mean_activation: torch.Tensor) -> None:
        """Initialise ``b_dec`` to the dataset-mean activation (standard trick).

        Call once before training with the mean over a sample of activations.
        """
        self.b_dec.data.copy_(mean_activation.to(self.b_dec.dtype))

    def save(self, path: str) -> None:
        """Save weights + the metadata needed to rebuild the module."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "d_in": self.d_in,
                "d_sae": self.d_sae,
                "l1_coeff": self.l1_coeff,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: torch.device | str = "cpu") -> "SparseAutoencoder":
        """Reconstruct an SAE saved with :meth:`save`."""
        ckpt = torch.load(path, map_location=device)
        sae = cls(d_in=ckpt["d_in"], d_sae=ckpt["d_sae"], l1_coeff=ckpt["l1_coeff"])
        sae.load_state_dict(ckpt["state_dict"])
        return sae.to(device)
