# XAI for Geospatial Foundation Models — Thesis Findings

Consolidated findings on applying XAI to Geospatial Foundation Models (GFMs) for land cover / crop classification. Scope: methods, feasibility per GFM access pattern, literature gaps, recommended workflow, and practical lessons from a hands-on Prithvi-600M experiment.

---

## TL;DR

1. **GFMs come in two access patterns.** *Type 1*: only pre-computed embeddings published (e.g. AlphaEarth). *Type 2*: backbone weights published (Prithvi, SatMAE, Tessera, TerraMind, Clay, DOFA). The pattern determines which XAI methods are even possible.
2. **Embedding-dim ranking is meaningless without binding.** Knowing "dim 12 matters" tells you nothing until you probe what dim 12 encodes (NDVI-peak-DOY, SWIR-winter, etc.).
3. **For Type 2, drop most embedding-level XAI.** Input-level methods (IG / AttnLRP / Occlusion) directly attribute to bands / dates / patches — interpretable in agronomic terms. Keep **probing classifiers** as the one embedding-level tool worth running: it makes structural claims about what the FM learned to represent.
4. **The thesis novelty is comparative XAI agreement under label scarcity.** No published work has compared GFM-classifier and raw-feature-classifier explanations on the same crop task across label budgets.
5. **Trust occlusion over gradient-based methods on ViT GFMs unless you've validated faithfulness.** In the hands-on Prithvi experiment, IG produced patch-grid artifacts; occlusion produced a physically plausible band-importance story.

---

## 1. The two GFM access patterns

| | **Type 1** | **Type 2** |
|---|---|---|
| Examples | AlphaEarth | Prithvi-EO-2.0, SatMAE, ScaleMAE, Tessera, TerraMind, Clay, DOFA |
| Published | Annual embedding tiles (e.g. 64-dim) | Backbone weights (`.pt` / `.safetensors`) |
| Gradient access | No | Yes |
| Attention access | No | Yes |
| Forward passes on new imagery | No (only the published embeddings) | Yes |

---

## 2. Method × access-pattern feasibility

| Method | Type 1 (embeddings only) | Type 2 (backbone available) |
|---|---|---|
| **Permutation importance** on classifier dims | Yes | Yes |
| **KernelSHAP / LIME** on classifier | Yes | Yes |
| **Per-dim ablation** (Benavides 2026) | Yes — primary tool | Yes |
| **Probing classifiers** for agronomic indices | Yes — critical binding step | Yes |
| **TCAV** (concept activation vectors) | Yes (at embedding layer) | Yes (any layer) |
| **Counterfactuals** in latent space | Yes | Yes |
| **Integrated Gradients** to bands/dates | No | Yes |
| **AttnLRP** (Achtibat 2024) | No | Yes — current SOTA for ViT |
| **Attention rollout** | No | Yes |
| **Occlusion** on input bands/dates/patches | No | Yes — most causal / trustworthy |
| **GradCAM** | No | CNN encoders only (rare in GFMs) |

---

## 3. Permutation importance vs SHAP — they're different

| | Permutation importance | SHAP |
|---|---|---|
| Question | "Does corrupting dim *i* hurt overall accuracy?" | "How did dim *i* push *this* prediction's score?" |
| Method | Shuffle dim *i*, re-evaluate, measure delta | Game-theoretic — average marginal contribution over feature subsets |
| Scope | Global only | Local (per prediction) and aggregable |
| Cost | Cheap | KernelSHAP expensive; TreeSHAP cheap (tree models only) |
| Handles correlated dims | Poorly — both look "moderately important" | Better — credit is split between correlated dims |

**Per-dim ablation** is *not* a third importance ranker — it's a **sufficiency test**: which minimum subset of dims preserves accuracy. Useful for (a) compressing the probing target, (b) revealing redundancy, (c) falsifying interpretation claims.

---

## 4. Literature gaps (thesis novelty positioning)

From a focused literature scan:

1. **No head-to-head XAI comparison across GFMs** on the same task. Every published paper looks at one model.
2. **GFM-vs-raw-feature explanation agreement is essentially unstudied.** Directly maps to CLAUDE.md research question #2. Strongest novelty claim.
3. **XAI under label scarcity** flagged as open by Höhl et al. 2024 and the Springer AI Review survey (10.1007/s10462-024-10803-5). Fits naturally with the 5 → 200 samples/class curves.
4. **Faithfulness evaluation on EO ViTs is rare.** Most papers stop at qualitative heatmaps; Quantus benchmarks are absent.
5. **Embeddings-only XAI is a young subfield.** Benavides et al. 2026 (arXiv:2603.16911 — verify ID before citing) is the first careful study, focused on AlphaEarth.

Two essential references for the methods toolbox:

- **Achtibat et al. 2024 — AttnLRP** (arXiv:2402.05602). Current SOTA faithful attribution for transformers; LXT library.
- **Hedström et al. 2023 — Quantus** (JMLR; github.com/understandable-machine-intelligence-lab/Quantus). 35+ metrics for faithfulness / robustness / complexity evaluation. Non-negotiable for a thesis.

Less foundational but useful:

- **Chefer et al. 2021** (CVPR) — Transformer Interpretability Beyond Attention Visualization.
- **Abnar & Zuidema 2020** — Attention rollout (baseline; brittle per Russwurm 2022).
- **Kakogeorgiou & Karantzalos 2021** — 10-method XAI benchmark on Sentinel multi-label; Occlusion, Grad-CAM, LIME most reliable.
- **Dantas et al. ECML 2023** — Sparse temporal counterfactuals for SITS classifiers.
- **Hall et al. 2024** (arXiv:2402.13791) — Systematic review of XAI in remote sensing; confirms GFM-specific XAI is essentially absent.

---

## 5. Recommended workflow

### If AlphaEarth (Type 1) is in scope

Run the full universal stack on every model. Embedding-level XAI is the only lens on Type 1, so for fair cross-model comparison every model gets the same treatment.

### If AlphaEarth is dropped (Type 2 only)

```
Default workflow:
  ├── Input-level XAI (the workhorse)
  │     ├── Integrated Gradients (Captum) — band × timestep attribution
  │     ├── AttnLRP (LXT, if model is supported) — ViT-faithful
  │     └── Occlusion — slow, gold-standard sanity check
  │
  ├── Probing classifiers (the one embedding-level method to keep)
  │     ├── Train linear regression: embedding → agronomic index
  │     │     (NDVI / EVI / NDWI per timestep, DOY of greenness peak,
  │     │      mean LAI, soil-moisture proxy)
  │     └── Report R² per index per model → structural claim about
  │           what each FM learned to represent
  │
  ├── Faithfulness evaluation (Quantus)
  │     ├── Faithfulness correlation
  │     └── Max-sensitivity
  │
  └── Comparison vs raw-feature baseline (the thesis novelty)
        ├── Same XAI applied to GFM-classifier and NDVI-RF-classifier
        ├── Spearman rank correlation on per-timestep importance
        └── Repeat across 5 / 10 / 20 / 50 / 100 / 200 samples/class

Skip on Type 2:
  ✗ SHAP-on-classifier per embedding dim
  ✗ Per-dim permutation importance
  ✗ Per-dim ablation (Benavides protocol)
  → All three duplicate input-level XAI with worse interpretability.
```

### Day-1 minimum viable run

1. One parcel set: 200 maize + 200 soy in the AOI.
2. Two classifiers: linear probe on FM embeddings; RF on raw NDVI time series.
3. Compute reference indices (NDVI/EVI/NDWI per timestep, DOY of peak) for the parcels.
4. Run IG + Occlusion (+ AttnLRP if available) on the FM side.
5. Probe FM embeddings against the reference indices.
6. Compare. Run Quantus.

End-of-day target: "FM embeddings encode NDVI-peak-DOY at R²=0.78; IG agrees August NIR dominates; raw-NDVI baseline picks the same timestep; agreement at 200 samples = 0.85."

---

## 6. Practical lessons from the Prithvi-600M hands-on

Setup: TerraTorch + Prithvi-EO-v2-600 + UNet decoder, RGB single-frame Dubai aerial LULC, frozen backbone, 6 classes including Water.

### IG was unreliable on this model

For the Water class:
- IG showed a **patch-grid artifact** (16×16 ViT token boundaries leaking into the heatmap).
- IG per-band ranking: RED ≈ BLUE ≈ GREEN, all small magnitudes.
- Occlusion per-band ranking: GREEN +520, BLUE −200, RED ≈ 0.
- Conclusion: **occlusion is the trustworthy signal**, IG is contaminated by ViT structure + integration noise.

Five reasons IG struggled here:
1. **ViT patch-grid leak** — uniform interpolation amplifies patch-edge gradients.
2. **Bad baseline** — zero in normalized space (= dataset mean image) is out-of-distribution for a pretrained EO ViT.
3. **Region-averaged forward** — gradient is a mean of per-pixel gradients, small and noisy.
4. **Domain mismatch** — Prithvi was pretrained on 6-band multispectral × multi-temporal; we fed it RGB × single frame.
5. **Decoder smearing** — UNet skip connections diffuse the gradient signal spatially.

### Fixes attempted

1. **fp32 cast + SmoothGrad via NoiseTunnel + n_steps=100** — improvement was marginal.
2. **AttnLRP via LXT** — failed: LXT's `monkey_patch` only supports a hardcoded list of architectures (Llama / Qwen / Gemma / Bert / GPT-2 / torchvision ViT). Prithvi isn't on the list. Making AttnLRP work on Prithvi requires writing a custom `patch_map` for Prithvi's attention class — a multi-hour side project.
3. **Input × Gradient (Captum `InputXGradient`)** — used as the pragmatic substitute: single forward + single backward, no integration path, no library compatibility issues. Less faithful than true AttnLRP but cleaner than vanilla IG on this model.

### Cross-method triangulation

For a publishable claim: run **at least three methods** and check agreement.
- All three agree → robust finding.
- Two agree, one disagrees → flag the outlier as unreliable on this model.
- All three disagree → the decision process is ambiguous; investigate further before claiming an explanation.

---

## 7. Conceptual reference

### Integrated Gradients (IG)

Post-hoc, inference-time method. Path integral of gradients from a baseline `x'` to the input `x`:

```
IG_i(x) = (x_i - x'_i) · ∫₀¹ ∂f(x' + α(x − x'))/∂x_i dα
```

Discretized by Captum into `n_steps` interpolations. Satisfies completeness (Σ IG_i = f(x) − f(x')). Failure modes: bad baseline choice, gradient saturation across many layers, ViT patch-grid leak.

### Attention rollout vs AttnLRP

Different methods.

- **Attention rollout** (Abnar 2020): multiply attention matrices across layers. No gradients, no class-discriminative target. Brittle.
- **AttnLRP** (Achtibat 2024): layer-wise relevance propagation with custom rules for softmax and Q·Kᵀ matmul. Class-discriminative, conservation-preserving. Implemented by monkey-patching `F.softmax` so that a normal `.backward()` call computes LRP relevance instead of the mathematical gradient. After backward, `input.grad × input` is the input-space relevance.

### Occlusion

The most "honest" XAI method: N+1 forward passes, hiding one feature each time. Direct causal measurement.
- Gradient-free → no saturation, no integration artifacts, no patch-grid leak.
- Slow (linear in number of features); fast for coarse features (bands, timesteps).
- Trustworthy baseline on ViT GFMs where gradient methods may be contaminated.

### Why embedding-dim ranking alone is meaningless

A dim index is a coordinate, not a feature. Without binding (probing classifier → "dim 12 encodes NDVI-peak-DOY at R²=0.78"), per-dim importance is uninterpretable. The only useful output of embedding-level XAI is the *bound* claim: "the model uses NDVI-peak-DOY, which lives in dim 12."

### Embedding-level vs input-level — different questions

| | Embedding-level XAI | Input-level XAI |
|---|---|---|
| Asks | What is the embedding *space* organized around? | What input drove *this* prediction? |
| Output | Structural — quantitative R² per agronomic index | Per-sample — band / date / patch attribution |
| Scale | Global, dataset-level | Local, per-input |
| Cost | Seconds | Minutes per image |
| Possible on Type 1 | Yes | No |

---

## 8. Toolchain

| Tool | Use | Notes |
|---|---|---|
| **Captum** | IG, SmoothGrad (NoiseTunnel), Occlusion, FeatureAblation, KernelSHAP, TCAV, Input × Gradient | Primary attribution library. Native PyTorch — works on any TerraTorch model after a tensor-returning wrapper. |
| **shap** | SHAP on downstream sklearn classifier | TreeExplainer fast for RF/XGBoost; KernelExplainer for SVM/LogReg. |
| **LXT** | AttnLRP for transformers | Only patches Llama / Qwen / Gemma / Bert / GPT-2 / torchvision ViT out of the box. Custom `patch_map` needed for Prithvi. |
| **Quantus** | Faithfulness / robustness / complexity metrics | Non-negotiable for a thesis. |
| **alibi** | Counterfactual prototypes | Latent-space CFs. |

Skip OmniXAI unless a dashboard is needed — Captum + SHAP + Quantus is leaner.

---

## 9. Open follow-ups

- Write a custom `patch_map` for Prithvi's attention class to unlock true AttnLRP.
- Implement the comparative-XAI agreement metric (Spearman rank correlation on per-timestep importance) for the label-scarcity sweep.
- Establish reference-signal pipeline (NDVI / EVI / NDWI / DOY-peak / LAI) for the probing classifier.
- Decide on baseline policy for occlusion across the thesis: zero (mean), random noise, or both reported.
