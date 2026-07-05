"""Practice pdb/ipdb on the real data shapes from ``activations.py`` — no Hugging Face.

Mimics the loop in ``src/sae_interp/activations.py::load_corpus_tokens`` but tokenises
real text with ``tiktoken``'s GPT-2 encoding — the *same BPE vocabulary* GPT-2 uses, so
the token ids are identical to what ``model.to_tokens(text)`` would return (we prepend
BOS, id 50256, to match the TransformerLens default). tiktoken downloads its vocab from
an OpenAI Azure blob (not huggingface.co, which is blocked) and caches it after the
first run.

What this can't replicate: the *activations* (needs GPT-2's weights → Hugging Face).
For those, harvest a small sample on Colab and commit it — see HANDOVER notes.

Run:
    python scripts/99_pdb_practice.py
"""

import torch

from sae_interp.config import load_config
from sae_interp.net import enable_os_trust_store

GPT2_BOS = 50256  # <|endoftext|>, prepended by model.to_tokens by default

# Real text, mixed lengths — like rows of the pile-10k corpus. The short ones tokenise
# to fewer than ctx_len (128) tokens and get skipped, exactly as in the real loop.
SAMPLE_TEXTS = [
    "I went to the bank to deposit a cheque before lunch.",
    (
        "The Golden Gate Bridge is a suspension bridge spanning the Golden Gate, the "
        "one-mile-wide strait connecting San Francisco Bay and the Pacific Ocean. The "
        "structure links the city of San Francisco to Marin County, carrying both U.S. "
        "Route 101 and California State Route 1 across the strait. It also carries "
        "pedestrian and bicycle traffic, and is designated as part of U.S. Bicycle "
        "Route 95. Recognised by the American Society of Civil Engineers as one of the "
        "Wonders of the Modern World, the bridge is one of the most internationally "
        "recognised symbols of San Francisco and California. At the time of its "
        "opening in 1937 it was both the longest and the tallest suspension bridge in "
        "the world."
    ),
    "Short snippet that will be skipped.",
    (
        "In computer science, a sparse autoencoder is a type of artificial neural "
        "network used to learn efficient codings of unlabelled data. The sparsity "
        "constraint forces the model to respond to the unique statistical features of "
        "the training data, so that most hidden units are inactive for any given "
        "input. In mechanistic interpretability research, sparse autoencoders are "
        "trained on the internal activations of a language model in order to decompose "
        "those activations into a larger dictionary of more interpretable features. "
        "Each learned feature ideally corresponds to a single human-understandable "
        "concept, such as a topic, a syntactic role, or a named entity, and the "
        "decoder weights indicate the direction that concept occupies in the model's "
        "residual stream."
    ),
    (
        "The quarterly report showed that revenue had increased by twelve per cent "
        "compared with the same period last year, driven primarily by strong growth in "
        "the small business insurance segment. Management attributed the improvement "
        "to better online conversion rates, a simplified quote journey, and expanded "
        "partnerships with brokers across the United Kingdom and the United States. "
        "Operating costs rose more slowly than revenue, so the underlying margin "
        "improved for the third consecutive quarter. The board declared an interim "
        "dividend and reiterated its full-year guidance, while cautioning that claims "
        "inflation and a softening rate environment remained the principal risks to "
        "the outlook for the remainder of the financial year. Analysts responded by "
        "modestly raising their price targets, and the shares closed three per cent "
        "higher on the day."
    ),
]


def to_tokens(enc, text: str) -> torch.Tensor:
    """Replicate ``model.to_tokens(text)``: (1, seq) int64 tensor with BOS prepended."""
    ids = [GPT2_BOS] + enc.encode(text)
    return torch.tensor(ids, dtype=torch.int64).unsqueeze(0)


def main() -> None:
    enable_os_trust_store()  # Netskope MITM cert — same fix the real scripts use
    import tiktoken  # imported lazily, after trust store injection

    enc = tiktoken.get_encoding("gpt2")  # GPT-2's actual BPE vocab (cached after run 1)
    cfg = load_config("configs/sae_gpt2small.yaml")
    ctx = cfg.data.ctx_len
    seqs: list[torch.Tensor] = []

    for i, text in enumerate(SAMPLE_TEXTS):
        toks = to_tokens(enc, text)
        breakpoint()  # <-- console opens here each iteration; try the commands below
        # (Pdb) toks.shape                          -> torch.Size([1, N])
        # (Pdb) toks.shape[1] < ctx                 -> will this row be skipped?
        # (Pdb) toks[0, :8]                         -> first 8 token ids (50256 = BOS)
        # (Pdb) enc.decode(toks[0, 1:9].tolist())   -> ...back to the original words
        # (Pdb) [enc.decode([t]) for t in toks[0, 1:9].tolist()]  -> one word per token
        # (Pdb) c                                   -> continue to the next text
        if toks.shape[1] < ctx:
            continue
        seqs.append(toks[0, :ctx])

    print(f"kept {len(seqs)} of {len(SAMPLE_TEXTS)} texts (needed >= {ctx} tokens)")
    print("stacked shape:", torch.stack(seqs).shape, "— (n_seq, ctx_len), as in activations.py")


if __name__ == "__main__":
    main()
