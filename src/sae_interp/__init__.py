"""sae_interp — a small, readable mechanistic-interpretability toolkit.

The package trains and analyses sparse autoencoders (SAEs) on the residual-stream
activations of a small language model (GPT-2-small by default), finds interpretable
features, and causally validates them.

Module map
----------
- ``config``        : load + validate the YAML config into a typed dataclass.
- ``device``        : pick the best available torch device (cuda -> mps -> cpu).
- ``activations``   : harvest + cache model activations at a hook point over a corpus.
- ``sae``           : the classic L1 ReLU sparse autoencoder.
- ``train``         : custom training loop (Adam, dead-feature resampling, logging).
- ``metrics``       : L0, reconstruction MSE, variance-explained, CE-loss-recovered, frontier sweep.
- ``features``      : max-activating examples, feature density, logit-lens naming.
- ``interventions`` : clamp / ablate features via TransformerLens hooks.
"""

__version__ = "0.1.0"
