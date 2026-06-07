"""Streamlit feature browser (stretch goal) — the "engineering edge" UI.

An interactive way to type a feature index, see its max-activating examples, density,
logit-lens tokens, and run a quick clamp/ablate intervention on a prompt. Loads the
model + pretrained SAE once and caches them across reruns.

Run:
    streamlit run app/feature_browser.py

Requires `pip install streamlit` (it's an optional dep). Needs Hugging Face access to
download the model + SAE the first time (see README "Network / corporate proxy").
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running via `streamlit run app/feature_browser.py` without installing the pkg.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402
import torch  # noqa: E402

from sae_interp import net  # noqa: E402

net.bootstrap()

from sae_interp.config import load_config  # noqa: E402
from sae_interp.device import get_device  # noqa: E402
from sae_interp import features as F, interventions as IV  # noqa: E402


@st.cache_resource
def load_everything(config_path: str):
    cfg = load_config(config_path)
    device = get_device()
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    model = HookedTransformer.from_pretrained(cfg.model.name, device=str(device))
    model.eval()
    sae = SAE.from_pretrained(cfg.pretrained_sae.release, cfg.pretrained_sae.sae_id, device=str(device))
    if isinstance(sae, tuple):
        sae = sae[0]
    return cfg, device, model, sae


def main() -> None:
    st.set_page_config(page_title="SAE Feature Browser", layout="wide")
    st.title("🔬 SAE Feature Browser — GPT-2-small")

    config_path = st.sidebar.text_input("Config", "configs/sae_gpt2small.yaml")
    with st.spinner("Loading model + SAE (first run downloads from Hugging Face)…"):
        cfg, device, model, sae = load_everything(config_path)
    st.sidebar.success(f"Loaded on {device} · d_sae={sae.cfg.d_sae}")

    feature_index = st.sidebar.number_input(
        "Feature index", min_value=0, max_value=int(sae.cfg.d_sae) - 1, value=0, step=1
    )

    tab_examples, tab_intervene = st.tabs(["Max-activating examples", "Causal intervention"])

    with tab_examples:
        st.subheader(f"Feature #{feature_index}")
        lens = F.logit_lens_tokens(model, sae, int(feature_index), top_k=10)
        st.write("**Logit-lens top tokens:**", ", ".join(f"`{t}`" for t, _ in lens))
        n_seq = st.slider("Corpus sequences to scan", 50, 1000, 200, step=50)

        if st.button("Find examples"):
            from sae_interp.activations import load_corpus_tokens

            with st.spinner("Scanning corpus…"):
                tokens = load_corpus_tokens(
                    _override_n(cfg, n_seq), model
                )
                acts = F.collect_feature_activations(
                    model, sae, tokens, cfg.model.hook_name, [int(feature_index)], device
                )
                density = F.feature_density(acts, 0)
                examples = F.max_activating_examples(model, acts, tokens, 0, top_k=15)
            st.metric("Feature density", f"{density:.3%}")
            for ex in examples:
                st.text(F.format_example(ex))

    with tab_intervene:
        prompt = st.text_input("Prompt", "I walked into the room and")
        mode = st.radio("Mode", ["clamp", "ablate"], horizontal=True)
        alpha = st.slider("Clamp strength α", 0.0, 30.0, 8.0) if mode == "clamp" else 0.0
        if st.button("Run intervention"):
            with st.spinner("Generating…"):
                res = IV.generate_with_intervention(
                    model, sae, prompt, cfg.model.hook_name, int(feature_index),
                    mode=mode, alpha=alpha, device=device,
                )
            col1, col2 = st.columns(2)
            col1.markdown("**Baseline**")
            col1.write(res.baseline_text)
            col2.markdown(f"**{mode.capitalize()}**")
            col2.write(res.intervened_text)


def _override_n(cfg, n_seq):
    cfg.data.n_sequences = int(n_seq)
    return cfg


if __name__ == "__main__":
    main()
