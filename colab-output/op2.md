[phase4] device=cuda  lambdas=[0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005]  steps/lambda=3000
config.json: 100% 665/665 [00:00<00:00, 2.17MB/s]
model.safetensors: 100% 548M/548M [00:04<00:00, 119MB/s]
Loading weights: 100% 148/148 [00:00<00:00, 6468.98it/s]
generation_config.json: 100% 124/124 [00:00<00:00, 557kB/s]
tokenizer_config.json: 100% 26.0/26.0 [00:00<00:00, 119kB/s]
vocab.json: 100% 1.04M/1.04M [00:00<00:00, 5.63MB/s]
merges.txt: 100% 456k/456k [00:00<00:00, 3.22MB/s]
tokenizer.json: 100% 1.36M/1.36M [00:00<00:00, 6.19MB/s]
Loaded pretrained model gpt2 into HookedTransformer
README.md: 100% 373/373 [00:00<00:00, 2.30MB/s]
dataset_infos.json: 100% 921/921 [00:00<00:00, 5.35MB/s]
tokenising corpus: 100% 2000/2000 [00:18<00:00, 110.57it/s]
harvesting blocks.8.hook_resid_pre: 100% 63/63 [00:15<00:00,  3.98it/s]
[activations] cached 256,000 activations to activation_cache/acts_db8693eb660b.pt

[phase4] === lambda = 1e-04 ===
[train] step      1/3000 | recon 1985291.500 | L1  1.2796 | L0 3070.8 | VE -27.690 | dead 0.0% |   2.2 it/s
[train] step   3000/3000 | recon  182.317 | L1  0.5890 | L0 1679.4 | VE 0.998 | dead 0.0% |  14.4 it/s
[train] done in 208.8s — saved results/checkpoints/sae_l11e-04_final.pt
[phase4] lambda 1e-04: L0=1678.9  VE=0.997

[phase4] === lambda = 2e-04 ===
[train] step   3000/3000 | recon  181.662 | L1  1.1950 | L0 1686.6 | VE 0.998 | dead 0.0% |  13.0 it/s
[train] done in 230.9s — saved results/checkpoints/sae_l12e-04_final.pt
[phase4] lambda 2e-04: L0=1686.1  VE=0.998

[phase4] === lambda = 5e-04 ===
[train] step   3000/3000 | recon  183.175 | L1  2.9886 | L0 1683.9 | VE 0.998 | dead 0.0% |  12.5 it/s
[train] done in 240.2s — saved results/checkpoints/sae_l15e-04_final.pt
[phase4] lambda 5e-04: L0=1683.6  VE=0.998

[phase4] === lambda = 1e-03 ===
[train] step   3000/3000 | recon  182.836 | L1  5.8760 | L0 1680.7 | VE 0.998 | dead 0.0% |  12.5 it/s
[train] done in 240.4s — saved results/checkpoints/sae_l11e-03_final.pt
[phase4] lambda 1e-03: L0=1680.1  VE=0.998

[phase4] === lambda = 2e-03 ===
[train] step   3000/3000 | recon  182.314 | L1 11.6323 | L0 1679.1 | VE 0.998 | dead 0.0% |  12.5 it/s
[train] done in 239.4s — saved results/checkpoints/sae_l12e-03_final.pt
[phase4] lambda 2e-03: L0=1678.7  VE=0.998

[phase4] === lambda = 5e-03 ===
[train] step   3000/3000 | recon  184.037 | L1 28.2145 | L0 1673.0 | VE 0.998 | dead 0.0% |  12.6 it/s
[train] done in 238.0s — saved results/checkpoints/sae_l15e-03_final.pt
[phase4] lambda 5e-03: L0=1672.5  VE=0.998
[metrics] saved frontier plot to results/figures/frontier.png

[phase4] wrote results/frontier_points.json and figures/frontier.png
This frontier plot is the headline figure for the writeup.

--- FRONTIER SUMMARY ---
lambda    L0      VE
1e-04   1678.9   0.997
2e-04   1686.1   0.998
5e-04   1683.6   0.998
1e-03   1680.1   0.998
2e-03   1678.7   0.998
5e-03   1672.5   0.998

Key finding: 50x increase in lambda (1e-4 → 5e-3) reduced L0 by only 6 points (1679→1673).
At 3000 steps, all lambda values are still in the reconstruction-learning phase — L1 penalty
has not yet differentiated. Sparsity requires sustained training (40k steps), not just
stronger lambda. The Phase 3 result (lambda=4e-3, 40k steps) achieved L0=1161, confirming this.
