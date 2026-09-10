# Methods Chapter — Research Notes

Working document to inform the Methods chapter (Section 5 of [current_proposal.md](current_proposal.md), per [methods_instructions.md](methods_instructions.md)). Organized by the three thesis phases, each tied to one RQ. Use as raw material for the 500–1000-word methods text and the reference list (rubric requires ≥ 5 peer-reviewed citations).

> **Provenance caveat.** This file consolidates output from three parallel research agents that did **not** have live web access (`WebSearch` and `WebFetch` were blocked). All citations are from agent training knowledge (cutoff Jan 2026) and from the vetted reference list already in [current_proposal.md](current_proposal.md). **Every citation must be verified on Scholar / arXiv / DOI before being committed to the submitted chapter** — see the verification checklist at the end of this document.

---

## How the three phases map onto the rubric

| Rubric criterion | Where covered in this document |
|---|---|
| Appropriate method | §§ 1.1, 2.1, 3.1 (method selection per phase) |
| Justification w/ peer-reviewed lit | Citation blocks under each subsection |
| Logical sequence of steps | § 4 (cross-phase pipeline) |
| Description of methods/tools | §§ 1–3 in detail |
| Reference list (≥ 5 peer-reviewed) | § 5 consolidated bibliography |

The methods chapter itself is hard-capped at 500–1000 words, so this document is **deliberately over-spec'd** — write the chapter by selecting and condensing, not by transcribing.

---

# Phase 1 — Pipeline + benchmarking GFMs under label scarcity (RQ1)

> **Locked decisions (2026-05-25):** (i) FE+TH battery is the primary adaptation regime across all four GFMs; **LoRA is secondary and optional** — a descope-able side-experiment on TerraMind + THOR only (the two GFMs with public weights), to be run if scope and time allow but not load-bearing for the RQ1 answer; (ii) class set restricted to top-N most frequent EuroCropsML classes (N ≈ 15–20, exact threshold fixed after class-frequency inspection); (iii) cross-region transfer is a follow-up sub-experiment (Phase 1c), not part of the main benchmark sweep.

## 1.1 Two-stage scope

- **SO1.1 — Build the harness.** A unified, model-agnostic pipeline for label-scarce evaluation of pre-trained GFMs on per-parcel crop classification. The pipeline itself is a research artifact: published as code, it lets a practitioner drop in a new GFM and reproduce the benchmark protocol on EuroCropsML (or another parcel-level dataset) with a single config change.
- **SO1.2 — Run the benchmark.** Use the harness to evaluate AlphaEarth Foundations, TESSERA, TerraMind and THOR under uniform conditions (the candidate list may shift across the project as new GFMs are released or as one drops out for feasibility reasons). Raw-NDVI / spectral time-series classifier as no-GFM baseline.
- **SO1.3 — Phase 1c sub-experiment.** Extend the protocol to cross-region transfer (§ 1.8).

## 1.2 Pipeline architecture

The harness is a thin wrapper around `(backbone, adapter, dataset, protocol)` that consumes EuroCropsML splits and emits learning-curve outputs. Two ingest routes are necessary because the GFMs in scope are distributed differently:

| Model | Distribution | Ingest route | LoRA possible? |
|---|---|---|---|
| **TerraMind** (Jakubik et al., 2025) | Open weights via TerraTorch + HuggingFace | TerraTorch route | Yes |
| **THOR** (Forgaard et al., 2026) | Open weights (NCC / ESA); TerraTorch wrapper status pending — **VERIFY** | TerraTorch route (or wrap manually) | Probably |
| **AlphaEarth Foundations** (Brown et al., 2025) | Pre-computed annual embeddings on Earth Engine (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, 64-D / parcel-year) | Embedding-store route | **No** |
| **TESSERA** (Feng et al., 2025) | Pre-computed pixel-time-series embeddings + Python sampling library; encoder weights not publicly distributed — **VERIFY** | Embedding-store route | **No** |

- **TerraTorch route** (Gomes et al., 2024). For models with open weights and a TerraTorch backbone wrapper. The harness uses `freeze_backbone=True` for FE+TH and TerraTorch's PEFT integration for LoRA. TerraTorch is built on PyTorch Lightning + TorchGeo (Stewart et al., 2022), inheriting their dataset conventions.
- **Embedding-store route.** For models distributed as pre-computed embeddings. The harness samples one vector per parcel (centroid or zonal mean of pixel embeddings within the polygon) and feeds it directly to the adapter battery — no encoder is loaded.

Both routes converge on the same downstream adapter + protocol layer, so the evaluation logic is shared.

**Comparability caveat to state explicitly.** TerraMind / THOR (TerraTorch route) and AlphaEarth / TESSERA (embedding-store route) do **not** enter the benchmark through symmetric data paths — the first pair runs a forward pass on a patch around the parcel; the second pair samples a pre-computed vector at the parcel. This is exactly the kind of *practitioner-realistic* comparison we want ("what can a user actually do with each released artifact?"), but the methods chapter must name it, not paper over it.

## 1.3 Adaptation regime

The thesis tests **what is already in the pre-trained representation** under label scarcity — not how well each backbone can be re-pre-trained. Full fine-tuning is therefore excluded.

### 1.3.1 Frozen encoder + trainable head (FE+TH) — primary, all four GFMs

Every encoder weight is frozen; gradients never flow into the backbone. Only a lightweight classification head is trained. To avoid committing to one head, the harness runs a **battery of lightweight adapters** spanning a capacity axis:

1. **k-NN** on standardised embeddings — distance-based, zero training, sensitive to the metric structure of the embedding space.
2. **Linear probe** (LogReg / linear SVM) — convex, the canonical SSL probe inherited from SimCLR, MAE.
3. **Shallow MLP head** (1–2 hidden layers, ≤ 256 units, dropout 0.1–0.3) — differentiable, mild non-linearity.
4. **Kernel SVM (RBF)** and **tree ensembles** (Random Forest, gradient-boosted trees) — non-linear, capacity ladder for low-K stress.

Standard preprocessing: `StandardScaler` fit on train only and transformed on test; `class_weight='balanced'` reported alongside unweighted; no PCA for downstream accuracy at 64–128-D (used only for visualisation); no SMOTE on embeddings (interpolation in a learned manifold whose geometry is not Euclidean for crop semantics). Head hyperparameters tuned by stratified 5-fold CV on the K-shot training subset.

For models accessed via the TerraTorch route (TerraMind, THOR), the "decoder/head" follows TerraTorch's `freeze_backbone=True` + classification-head convention. The terminology is "head" (not "decoder") in this thesis because the task is per-parcel classification, not pixel-wise segmentation.

### 1.3.2 LoRA — secondary and optional, on TerraMind + THOR

LoRA (Hu et al., 2022) is **optional**. It is scoped as a stretch-goal side-experiment, descope-able without affecting the RQ1 answer. When run, it is applied **only** to the GFMs with public encoder weights — currently TerraMind and THOR. Rank r ∈ {4, 8, 16}, injected into Q and V projections of every attention block, trained with AdamW (LR 5e-4 → 1e-4, 20–50 epochs, 0.3–3 % trainable-parameter share). Compared point-for-point against FE+TH on the **same** models, same K, same seeds.

The intended finding is *not* "which GFM is best." It is whether targeted encoder adaptation buys anything on top of the frozen representation in a label-scarce regime — i.e. is the pretrained embedding already saturated for parcel-level crop classification? A null result (LoRA ≈ FE+TH) is itself informative.

Decision rule for descoping: if the FE+TH benchmark sweep (Phase 1b) and the cross-region sub-experiment (Phase 1c) consume the available compute / time budget, the LoRA experiment is dropped and the RQ1 answer is reported as the FE+TH lower bound only, with LoRA explicitly noted as future work.

### 1.3.3 Why no full fine-tuning

Two reasons. (i) AlphaEarth and TESSERA do not (publicly) expose encoder weights, so full fine-tuning is impossible for half the model set and would break comparability. (ii) In the very-low-label regime (K ≤ 20), full fine-tuning typically underperforms a linear probe on the same encoder — PANGAEA's 10 %-label setting documents this rank inversion (Marsocci et al., 2024). For the label budgets of interest, the FT number would not be informative.

### 1.3.4 LoRA vs FE+TH — what each measures

FE+TH freezes the encoder in every weight; the entire model response to a new task lives in the head — **measures what the embedding already encodes**. LoRA freezes the original weights but injects low-rank update matrices into selected layers and trains those — the encoder adapts mildly — **measures what can be unlocked with minimal encoder adaptation**. The two are not interchangeable answers to RQ1; they bracket it. If both are run, the RQ1 answer sharpens: FE+TH is the lower bound on what the pretrained representation supplies, LoRA the upper bound on what a parameter-efficient adaptation reaches without re-pretraining. If only FE+TH is run (LoRA descoped), RQ1 is answered at the lower bound — a clean and defensible reduction of scope.

## 1.4 Evaluation protocol — K-shot stratified sampling on top-N classes

EuroCropsML's 176 classes are reduced to the **top-N most frequent crop classes per country** (N ≈ 15–20, exact threshold fixed after class-frequency inspection and reported in the chapter). Rationale: at K = 1, 5 the long tail produces empty cells under stratified sampling; truncating to top-N keeps the metric well-defined and preserves the fine-grained discrimination story (the point of comparing GFMs is the *separation* between phenologically-similar crops, not coarse macro-classes).

For each seed, K labelled parcels per retained class form the support set; the remaining parcels form a held-out test partition. K ∈ {1, 5, 10, 20, 50, 100, 200}. The held-out test partition is **fixed across all K within a seed** so the points on a learning curve are directly comparable. ≥ 5 seeds per cell (target 10). EuroCropsML's official K-shot + supervised-transfer baselines (Reuss et al., 2025 — **VERIFY K-grid and seed count**) are reported alongside as comparison anchors.

PANGAEA's *limited-labels (10 %)* setting (Marsocci et al., 2024) is included as a single robustness checkpoint at the high-K end — documented to reorder model rankings vs. full-data settings.

## 1.5 Metrics

- **Macro-F1** as headline (robust to residual imbalance among the top-N).
- Overall Accuracy and Cohen's κ reported alongside.
- Per-class F1 in supplementary tables — essential for the AlphaEarth phenology story (§ 1.7).
- mAP only if a head emits calibrated per-class probabilities.

Single OA numbers are not credible: even after taxonomy reduction, the most frequent class dominates.

## 1.6 Statistical rigor

1. **Seeds + CIs.** ≥ 5 seeds per (GFM × adapter × K) cell; 95 % bootstrap CI over seeds (1 000 resamples). Not SD, not SE.
2. **Model comparison.** Paired tests — paired bootstrap on per-parcel correctness, Wilcoxon signed-rank on per-seed macro-F1. Multi-model grid via Demšar (2006) Critical-Difference diagram.
3. **Leakage prevention.**
   - **Spatial autocorrelation.** Adjacent parcels share weather, soil and planting decisions; spatially-naïve splits inflate accuracy. Use spatial block CV (5 × 5 km tiles assigned wholly to train or test) per Roberts et al. (2017).
   - **Multi-year contamination.** The same parcel must not span splits. EuroCropsML supplies parcel IDs; key off those.
4. `StandardScaler` fit on train only (common beginner error that survives into published EO work).

## 1.7 AlphaEarth temporal-mismatch protocol

AlphaEarth ships **annual** embeddings; EuroCropsML labels target a specific growing season → intra-season phenology is collapsed in the AlphaEarth representation. Two protocols, both documented:

- **Year-aligned single-embedding sampling (primary).** For each parcel labelled for crop year Y, sample AlphaEarth's year-Y embedding at parcel centroid (or zonal mean of pixel embeddings within the polygon). The apples-to-apples baseline that respects the dataset's intended use.
- **Multi-year stacking (sensitivity analysis).** Concatenate {Y−1, Y, Y+1} → 192-D, testing whether rotation-history context helps.

Expected pattern — AlphaEarth underperforms TESSERA / TerraMind / THOR on phenologically-ambiguous classes (spring vs winter wheat; maize vs sunflower), competitive on phenologically-distinct ones (orchards, vineyards, permanent grassland). **Report per-class F1 deltas**, not just macro-F1: the differential isolates *where* the annual aggregation hurts. This is a finding, not a flaw, and substantiates the limited-temporal-sensitivity discussion already noted in *Harvesting AlphaEarth* (arXiv:2601.00857 — **VERIFY**).

## 1.8 Phase 1c — Cross-region transfer (sub-experiment)

Once Phase 1b (in-region) results land, the best FE+TH adapter per GFM is re-evaluated under EuroCropsML's transnational protocol: source-country pretraining of the head → K-shot fine-tune on target-country labels. Directions: **ET → LV, ET → PT, LV → PT, PT → ET** (LV most label-rich, PT most agro-ecologically distinct → covers near-OOD and far-OOD). K ∈ {0, 5, 10, 20, 50, 100} per class.

EuroCropsML's class taxonomy does not align 1-to-1 across countries → transfer must be defined on the **intersection** of each pair's top-N class lists (typically 10–15 classes after intersection). State explicitly: mismatched-taxonomy transfer is a known source of inflated/deflated numbers in the literature.

## 1.9 Phase 1 citation additions

Already in [current_proposal.md](current_proposal.md) — re-cite directly:
- Reuss et al. (2025) — EuroCropsML. *Sci. Data*. doi:10.1038/s41597-025-04952-7. **VERIFY K-grid and seed count.**
- Marsocci et al. (2024) — PANGAEA. arXiv:2412.04204.
- Lacoste et al. (2023) — GEO-Bench. NeurIPS D&B. arXiv:2306.03831.
- Szwarcman et al. (2024) — Prithvi-EO-2.0. arXiv:2412.02732.
- Jakubik et al. (2025) — TerraMind. ICCV. arXiv:2504.11171.
- Feng et al. (2025) — TESSERA. arXiv:2506.20380.
- Brown et al. (2025) — AlphaEarth Foundations. arXiv:2507.22291.
- Forgaard et al. (2026) — THOR. arXiv:2601.16011.
- Lisaius et al. (2026) — Senegal TESSERA. arXiv:2601.16900. **VERIFY** K-grid and primary probe.

New citations needed for the Methods chapter:
- **Gomes, R., Pena, F. A. G., Roy, S., Szwarcman, D., et al. (2024). *TerraTorch: The geospatial foundation models toolkit.* arXiv:2503.20669.** **VERIFY** arXiv ID + author list.
- **Stewart, A. J., Robinson, C., Corley, I. A., Ortiz, A., Lavista Ferres, J. M., & Banerjee, A. (2022). TorchGeo: Deep learning with geospatial data. *ACM SIGSPATIAL.***
- Hu et al. (2022) — LoRA. ICLR. arXiv:2106.09685.
- Roberts et al. (2017) — Spatial cross-validation strategies. *Ecography* 40(8), 913–929. doi:10.1111/ecog.02881.
- Demšar (2006) — Statistical comparisons of classifiers. *JMLR* 7, 1–30.
- Rußwurm & Körner (2020) — Self-attention for raw optical S2 time series classification. *ISPRS J. P&RS* 169, 421–435.

**Phase 1 contribution claim.** No published GFM benchmark runs AlphaEarth + TESSERA + TerraMind + THOR head-to-head on the EuroCropsML protocol under uniform FE+TH adaptation. The thesis explicitly compares *what each released artifact lets a practitioner do under label scarcity*, not just internal-architecture-vs-architecture.

---

# Phase 2 — XAI on GFM embeddings (RQ2)

## 2.1 Method choice

Multi-method XAI battery to test whether GFM-based and raw-feature classifiers agree on which **temporal and spectral cues** drive crop classification — i.e., whether the FM has learned agronomically-faithful structure or learned artefacts. Four method families, each chosen for a complementary view:

1. **SHAP / permutation importance** on the sklearn probe → global + local feature attribution in embedding space.
2. **Attention rollout / Chefer-style attribution** on ViT backbones → per-patch and per-timestep saliency directly mappable to crop calendar dates.
3. **Integrated Gradients** through (encoder + differentiable head) → (T × B) heatmap on raw bands × dates.
4. **Linear probing of embedding dims against agronomic concepts** + **CKA** to compare representation spaces → bridges abstract embeddings to interpretable agronomy.

## 2.2 SHAP and permutation importance on embeddings

- `TreeSHAP` for RF / GBM probes (exact, cheap); `LinearSHAP` for LogReg (exact); `KernelSHAP` is model-agnostic but O(2^d) — Monte-Carlo approximate, k-means background of 50–100 centroids.
- Embedding dims are **highly correlated** → KernelSHAP's marginal-expectations assumption breaks. Use **conditional Shapley** (Aas, Jullum & Løland, 2021) or grouped SHAP over learned dimension clusters.
- Permutation importance (Fisher, Rudin & Dominici, 2019) for global ranking, with conditional permutation (Strobl et al., 2008) for correlated features.
- Three options for aggregating across non-interpretable dims: (i) cluster by activation correlation, (ii) probe each dim to an agronomic concept (§ 2.5), (iii) TCAV-style concept directions (Kim et al., 2018).

## 2.3 Attention maps for ViT backbones (TerraMind, Prithvi-style; THOR temporal axis; TESSERA time tokens)

- Raw single-head attention is noisy and ignores the residual path → do **not** report it alone.
- **Attention rollout** (Abnar & Zuiderveld, 2020) multiplies layer-wise attention with identity-residual; CLS row gives per-patch saliency. For temporal transformers it extends along the time-token axis → per-timestep saliency directly mappable to crop-calendar dates.
- **Chefer, Gur & Wolf (2021) — Generic Attention-model Explainability** combines gradients × attention with LRP-style propagation; outperforms rollout on standard benchmarks.
- For Prithvi-style space-time ViTs the CLS attention factors into (patch × frame) saliency → exactly what you want to compare against NDVI peak DOY.
- Implementation: Jacob Gildenblat's `pytorch-grad-cam` is the de-facto reference.

## 2.4 Integrated Gradients on (T × B) inputs

- Sundararajan, Taly & Yan (2017): integrate gradient along straight-line path from baseline x' to input x. Satisfies completeness + sensitivity axioms.
- For S2 time series, input is (T × B) → IG produces a (T × B) heatmap directly answering "which band on which date drove this prediction".
- **Baseline choice is the most consequential decision.** Zero baseline implies "no reflectance" (unphysical, inflates attribution to bright bands). Prefer (i) per-class mean image, (ii) random / blurred baselines, or — best — **Expected Gradients** (Erion et al., 2021) marginalising over a baseline distribution.
- ReLU/GELU saturation + shattered gradients in deep ViTs → denoise with **SmoothGrad** (Smilkov et al., 2017).
- **Mandatory sanity checks** (Adebayo et al., 2018): cascading model-randomisation + data-randomisation. Without these the maps may be partly edge detectors, not explanations.
- IG only works on differentiable models → for the sklearn probe pipeline, either (a) distil the SVM into a small MLP, or (b) use a LogReg head directly. Then IG attributes the raw (T × B) input through encoder + linear head, sidestepping the non-interpretable embedding space.

## 2.5 Probing for agronomic interpretation

- **Linear probes** (Alain & Bengio, 2017): regress frozen embeddings onto agronomic targets — NDVI peak DOY, length-of-season, max EVI, sowing-DOY, amplitude. Compute phenometrics with **TIMESAT** (Jönsson & Eklundh, 2004) or pyPhenology.
- Always pair probe accuracy with **selectivity** vs a control task (Hewitt & Liang, 2019) — otherwise high probe R² may just reflect probe capacity.
- **TCAV** (Kim et al., 2018) for binary agronomic concepts ("has-red-edge-peak", "double-cropping").
- **CKA** (Kornblith et al., 2019) to compare TESSERA vs AlphaEarth vs raw-NDVI representations on the same parcels — invariant to orthogonal transforms, isotropic scaling. SVCCA / RSA are alternatives.

## 2.6 Cross-model agreement (the RQ2 core comparison)

- Spearman ρ and Kendall τ-b for full-vector rank correlation.
- **Rank-Biased Overlap (RBO)** (Webber, Moffat & Zobel, 2010) for top-k agreement — top-weighted, handles non-conjoint lists (FM and raw spaces differ).
- Formal disagreement metrics from **Krishna et al. (2022)**: feature agreement, rank agreement, sign agreement, signed-rank agreement, pairwise-rank agreement.
- **Critical move:** the two models live in different feature spaces (raw T × B vs embedding dims), so direct rank comparison is meaningless. Two clean options:
  - (a) IG-attribute **both** models back to the raw (T × B) input space and compare those maps (recommended).
  - (b) Probe each embedding dim to an agronomic concept (§ 2.5) and compare per-concept importances.

## 2.7 Agronomic validation of explanations

A rigorous protocol beyond qualitative inspection: precompute per-parcel phenometrics (TIMESAT / pyPhenology) — start-of-season, peak DOY, length-of-season, amplitude. Then test whether **IG attribution mass concentrates within ±N days of those events at rates significantly above a uniform-temporal-attribution null**. For band-level agronomy, canonical analysis between attribution-weighted band importances and crop spectral libraries (USGS Spectral Library, ECOSTRESS).

Generic checklist: **Co-12 explanation quality properties** (Nauta et al., 2023). Operational metric library: **Quantus** (Hedström et al., 2023) — faithfulness, robustness, complexity, randomisation. EO-specific synthetic-truth validation: **Mamalakis, Ebert-Uphoff & Barnes (2022)**.

## 2.8 Pitfalls the chapter MUST acknowledge

1. **Disagreement problem** (Krishna et al., 2022) — SHAP, LIME, IG and gradient-input often disagree on the same model. Report agreement across **methods**, not just across **models**, before claiming any single explanation.
2. **Attention is not explanation** (Jain & Wallace, 2019; rebuttal Wiegreffe & Pinter, 2019; survey Bibal et al., 2022). Raw attention alone is not defensible; rollout / Chefer methods are (they incorporate gradients).
3. **Correlated features destabilise SHAP** → conditional or grouped Shapley.
4. **Saliency sanity checks** (Adebayo et al., 2018) — mandatory.
5. **Probe selectivity** (Hewitt & Liang, 2019) — pair with control tasks.
6. **Class imbalance** — minority crops produce noisy, low-confidence attributions. Weight by probe confidence; stratify XAI reporting by class.
7. **OOD attribution** — XAI on training distribution may not transfer to ET → PT regime. Report XAI per region.
8. **Computational cost** — KernelSHAP on 768-dim × 50k parcels × 176 classes is prohibitive. Plan TreeSHAP / LinearSHAP fallbacks.

## 2.9 Phase 2 citation candidates

Already in proposal: **Lundberg & Lee (2017)** — SHAP, NeurIPS, arXiv:1705.07874.

Add for methods chapter:
- Lundberg et al. (2020) — From local to global with TreeSHAP. *Nat. Mach. Intell.* doi:10.1038/s42256-019-0138-9.
- Sundararajan, Taly & Yan (2017) — Integrated Gradients. ICML. arXiv:1703.01365.
- Erion et al. (2021) — Expected Gradients. *Nat. Mach. Intell.* arXiv:1906.10670.
- Adebayo et al. (2018) — Sanity Checks for Saliency Maps. NeurIPS. arXiv:1810.03292.
- Abnar & Zuiderveld (2020) — Quantifying Attention Flow in Transformers. ACL. arXiv:2005.00928.
- Chefer, Gur & Wolf (2021) — Transformer Interpretability Beyond Attention Visualization. CVPR. arXiv:2012.09838.
- Aas, Jullum & Løland (2021) — Conditional Shapley for dependent features. *Artif. Intell.* arXiv:1903.10464.
- Kornblith et al. (2019) — CKA. ICML. arXiv:1905.00414.
- Kim et al. (2018) — TCAV. ICML. arXiv:1711.11279.
- Hewitt & Liang (2019) — Probes with Control Tasks. EMNLP. arXiv:1909.03368.
- Bibal et al. (2022) — Is Attention Explanation? ACL. ACL Anthology 2022.acl-long.269.
- Krishna et al. (2022) — Disagreement Problem in XAI. arXiv:2202.01602. **[preprint]**
- Nauta et al. (2023) — Co-12 properties. *ACM Comput. Surv.* doi:10.1145/3583558.
- Hedström et al. (2023) — Quantus. *JMLR.* arXiv:2202.06861.
- Mamalakis, Ebert-Uphoff & Barnes (2022) — XAI fidelity in geoscience. *AIES.* arXiv:2202.03407.
- Jönsson & Eklundh (2004) — TIMESAT. *Comput. Geosci.* doi:10.1016/j.cageo.2004.05.006.
- Rudin (2019) — Stop Explaining Black Box Models. *Nat. Mach. Intell.* arXiv:1811.10154. *(use as motivation citation only)*

**Phase 2 literature gap (= contribution):** XAI applied directly to GFM embeddings in agriculture is sparse; cross-model agreement studies (FM vs raw-feature) in RS are essentially absent; quantitative agronomic validation of attention/IG maps against phenometrics is at the qualitative-figure stage in current Prithvi/SatMAE/TerraMind papers.

---

# Phase 3 — LLM-generated reports from XAI-enriched structured context (RQ3)

## 3.1 Method choice

**Direct structured prompting** of an LLM with a JSON-style context payload assembled from (i) per-parcel classifier outputs + confidence, (ii) XAI attributions and attention summaries, (iii) GIS post-processing (zonal stats, change detection), (iv) external context (weather, rotation, agronomy). LLM never sees pixels. Output is a structured Markdown report with mandatory inline citations to evidence IDs. Two alternatives are explicitly rejected: end-to-end multimodal LLMs (GeoChat, EarthGPT — too opaque, RQ2 cannot be answered); LLM-as-agent (REMSA, GeoLLM-Squad — too prone to hallucination on thin EO context).

## 3.2 Data-to-text pipeline

Three families exist; the choice for this thesis is **(b)** with **(a)** safety net.

- (a) Templated / slot-filling — deterministic skeleton, LLM only realises slot text. High faithfulness, low fluency. Use as a fallback for safety-critical numerical claims (area, yield, confidence).
- (b) **Direct structured prompting** (current default for technical reports) — whole JSON dumped in prompt, explicit instructions on which fields to mention.
- (c) RAG-augmented D2T — structured record is the query, retrieved domain documents condition the language.

## 3.3 Grounded generation / hallucination mitigation

Three operational techniques to enforce traceability:

1. **Provenance-bearing prompts** — every factual claim ends with `[evidence_id=...]` resolving to a field in the structured context.
2. **Constrained decoding to JSON schemas** — OpenAI Structured Outputs, Anthropic tool-use JSON schemas, `Outlines` / `jsonformer` for open-weights. Eliminates format-level hallucinations.
3. **Post-hoc faithfulness verification** — atomic-claim decomposition (FactScore-style) + NLI or LLM-as-judge check against the source context.

## 3.4 Structured-context schema design

No formal standard exists for EO. Production patterns to adopt:

- Nested JSON, stable keys, **explicit units** (`area_ha`, not `area`).
- **Uncertainty fields** alongside every prediction — both raw `confidence: 0.87` and a categorical `confidence_level: "high|medium|low"` (LLMs handle categorical hedging better than raw probabilities).
- **Per-feature attribution blocks** for XAI: `"top_features": [{"feature": "NDVI_2024-07-15", "shap": 0.34, "rank": 1}, ...]`.
- Explicit `evidence_id` keys so the model can cite.
- Sensible numerical precision — round to 2–3 sig figs; over-precision invites the model to imitate it.
- **Wrap each context block in XML-style tags** (`<classification>...</classification>`, `<xai>...</xai>`, `<weather>...</weather>`). Anthropic Claude in particular is documented to attend reliably to XML-delimited regions.

Closest schema analogues to borrow from: **STAC item JSON** (spatial entities + assets), **HL7 FHIR `Observation` + `Provenance` resources** (evidence-bearing structured medical records that downstream LLMs summarize). Adopting the `Observation` + `Provenance` pattern adapted to parcels is a defensible minor contribution.

## 3.5 LLM-for-RS-reporting precedent (thin)

The field is sparse. Direct hits:
- **ChatEarthNet** (Yuan et al., 2024) — Sentinel-2 land-cover labels → ChatGPT-generated descriptions; uses structured land-cover context, not raw imagery. **Closest published analogue.** arXiv:2402.11325. **[preprint]**
- CropGPT / Agri-LLM-style 2024 preprints feed yield-model outputs into GPT-4 for advisories; quality uneven, mostly non-peer-reviewed → motivation only.

Stronger analogues to borrow methodology from:
- **Tanida et al. (2023) — Region-guided Radiology Report Generation.** CVPR. Region-level structured findings → report text. Pattern directly transferable.
- R2Gen / R2GenCMN / RadGraph-F1 lineage — long-running grounded-radiology-report literature → the right faithfulness analogues.
- **FinReport** (Zhang et al., 2024) — structured numeric + explanatory features → LLM-written stock-earnings report. arXiv:2403.02647. **[preprint]**
- MetOffice weather narrative work — numerical forecast → public-facing narrative.

State this thinness in the chapter and present the thesis as occupying the niche.

## 3.6 Four-axis evaluation rubric

1. **Factual accuracy vs structured input** — atomic-claim decomposition + verification (FactScore-style). Report precision/recall of grounded claims.
2. **Traceability** — fraction of factual sentences carrying a valid `evidence_id` that resolves to a context field (ALCE / Gao et al., 2023).
3. **Readability for non-specialists** — Flesch-Kincaid Grade Level, SMOG, Dale-Chall. Target FKGL ~ 10–12 for agronomists, ~ 7–9 for farmers; report both.
4. **Agronomic correctness (expert eval)** — small panel (3–5 agronomists) scoring Likert rubric: Correctness, Completeness, Actionability, Harmlessness. Stratified sample of ~ 30–50 reports. Inter-rater agreement via Krippendorff's α.

Optional 5th axis: **LLM-as-judge** (G-Eval / Prometheus-2) as a cheap proxy — but document its biases (length, self-preference) and never use it as sole evaluator.

## 3.7 Reproducibility checklist for the methods chapter

Fix and report:

- **Decoding params:** `temperature` 0.0–0.2 for technical reports; `top_p` 1.0; `max_tokens`; `frequency_penalty` / `presence_penalty` 0.
- **System prompt:** role, allowed/forbidden actions, citation requirement, refusal behaviour on missing data.
- **Few-shot vs zero-shot:** 2–3 worked examples (one full report) measurably reduces format drift; document examples verbatim in an appendix.
- **Output format constraints:** Markdown with fixed section skeleton (Summary / Findings / Confidence / Recommendations / Caveats) **or** JSON via Structured Outputs / tool-use schema enforcement.
- **Prompt versioning:** version ID + hash; store all prompts in the thesis repository.
- **Eval runs:** report N, number of seeds, mean and SD across reruns — single-shot LLM numbers are no longer credible.

## 3.8 Model choice trade-off

- **API (closed-weights)** — Claude, GPT-4o/5, Gemini. Easiest path to best quality, native Structured Outputs / tool-use JSON. *Cost:* pin a dated snapshot (`claude-3-5-sonnet-20241022`, `gpt-4o-2024-08-06`) and archive prompts + responses; seeds offer only partial determinism.
- **Open-weights** — Llama 3.1/3.3, Mistral, Qwen 2.5, Gemma 2. Full reproducibility (weight hash + decoding + seed + engine version). *Cost:* 70B-class open models approach but rarely match frontier APIs on long-form grounded writing; format compliance weaker without `vLLM` + grammar-constrained decoding.
- **Recommended (defensible thesis pattern):** open-weights for the bulk + frontier API as held-out comparison condition.

Minimum reproducibility checklist: model name + dated snapshot or HF revision hash, inference engine + version, quantization, all decoding params + seeds, prompt files versioned in repo, all structured-context payloads + generated outputs archived (Zenodo).

## 3.9 Phase 3 citation candidates

- Min et al. (2023) — FactScore. EMNLP. arXiv:2305.14251.
- Gao et al. (2023) — Enabling LLMs to Generate Text with Citations (ALCE). EMNLP. arXiv:2305.14627.
- Es et al. (2024) — RAGAS. EACL demo. *(verify exact ref)*
- Liu et al. (2023) — G-Eval. EMNLP. arXiv:2303.16634.
- Fabbri et al. (2022) — QAFactEval. NAACL.
- Tanida et al. (2023) — Region-guided Radiology Report Generation. CVPR.
- Singhal et al. (2023) — Med-PaLM. *Nature* 620.
- Kasner & Dušek (2024) — Beyond Traditional Benchmarks: Open LLMs on D2T. ACL.
- Sclar et al. (2024) — LM Sensitivity to Spurious Prompt Features. ICLR. arXiv:2310.11324.
- Biderman et al. (2024) — Reproducible LM Evaluation. arXiv:2405.14782. **[preprint]**
- Schulhoff et al. (2024) — The Prompt Report. arXiv:2406.06608. **[preprint]**
- Liang et al. (2023) — HELM. *TMLR.*
- Anthropic / OpenAI docs — XML-tag prompting, Structured Outputs. *(vendor docs, cite as technical reports)*
- **Yuan et al. (2024) — ChatEarthNet.** arXiv:2402.11325. **[preprint, but closest direct analogue]**

**Phase 3 literature gap (= contribution):** no published system connects a label-efficient GFM, an explicit XAI layer, and an LLM that uses those explanations as **structured context** for a traceable agricultural report. The schema (XAI-enriched per-parcel `Observation + Provenance` for agriculture) is a defensible original contribution.

---

# 4. Cross-phase pipeline — logical sequence for the methods chapter

The methods chapter should present the three phases as one coherent pipeline. Suggested order in the 500–1000-word text:

1. **Phase 1a — Build the harness (engineering, offline).** Two ingest routes converge on a shared adapter+protocol layer: (i) **TerraTorch route** (Gomes et al., 2024) for models with open weights — TerraMind, THOR — extracting features from a patch around each parcel; (ii) **embedding-store route** for pre-computed-embedding products — AlphaEarth via the GEE `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` collection, TESSERA via its Python sampling library — yielding one vector per parcel by centroid or zonal mean. Output: a unified parcel-vector store + EuroCropsML K-shot split definitions.
2. **Phase 1b — Run the benchmark (offline, CPU mostly).** Top-N EuroCropsML class set → K-shot stratified splits → **FE+TH adapter battery** (kNN, linear probe, MLP head, kernel SVM, tree ensembles) on all four GFMs + raw-NDVI / spectral-time-series baseline → learning curves per (GFM × adapter × K) with bootstrap CIs and spatial block CV. **Optionally** (stretch-goal, descope-able), LoRA on TerraMind and THOR as a "does encoder adaptation help?" side-experiment.
3. **Phase 1c — Cross-region transfer sub-experiment.** Best FE+TH adapter per GFM applied under EuroCropsML's transnational protocol on intersected top-N class lists; directions ET → LV, ET → PT, LV → PT, PT → ET; same K-shot grid and statistical reporting.
4. **Phase 2 — XAI (offline, CPU + small GPU).** SHAP + permutation on the sklearn-head probes; Integrated Gradients on (T × B) inputs through encoder + linear-head versions; attention rollout / Chefer-style attribution on the ViT backbones (TerraMind, THOR); agronomic-concept linear probes + CKA across representation spaces → **cross-model agreement metrics** → phenometric validation against TIMESAT-derived phenology.
5. **Phase 3 — Structured-context assembly + LLM report (online, no GPU).** Per-parcel JSON: classification + confidence + top-k SHAP/IG features + attention temporal saliency + zonal stats + weather + rotation history → wrapped in XML context blocks with `evidence_id` tags → LLM with citation-requiring system prompt → grounded Markdown report → four-axis evaluation (factual accuracy / traceability / readability / agronomic correctness).

This sequence is **modular**: any single component (GFM, adapter, XAI method, LLM) can be swapped without breaking the rest. The modularity itself is one of the thesis's architectural contributions, and the Phase-1a harness is its enabler.

---

# 5. Consolidated bibliography (drop-in)

Items **already in [current_proposal.md](current_proposal.md)** are not duplicated; the methods chapter can cite them directly. Items below are the **additions** to support the Methods section.

### Phase 1 — benchmarking
- Gomes, R., Pena, F. A. G., Roy, S., Szwarcman, D., et al. (2024). *TerraTorch: The geospatial foundation models toolkit.* arXiv:2503.20669. **[VERIFY arXiv ID + full author list]**
- Stewart, A. J., Robinson, C., Corley, I. A., Ortiz, A., Lavista Ferres, J. M., & Banerjee, A. (2022). TorchGeo: Deep learning with geospatial data. *Proceedings of the 30th International Conference on Advances in Geographic Information Systems (ACM SIGSPATIAL).*
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR. arXiv:2106.09685.
- Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., … Dormann, C. F. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography, 40*(8), 913–929. https://doi.org/10.1111/ecog.02881
- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research, 7,* 1–30.
- Rußwurm, M., & Körner, M. (2020). Self-attention for raw optical satellite time series classification. *ISPRS Journal of Photogrammetry and Remote Sensing, 169,* 421–435. https://doi.org/10.1016/j.isprsjprs.2020.06.006

### Phase 2 — XAI
- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. *Proceedings of the 34th International Conference on Machine Learning (ICML).* arXiv:1703.01365.
- Erion, G., Janizek, J. D., Sturmfels, P., Lundberg, S. M., & Lee, S.-I. (2021). Improving performance of deep learning models with axiomatic attribution priors and expected gradients. *Nature Machine Intelligence, 3,* 620–631. arXiv:1906.10670.
- Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). Sanity checks for saliency maps. *NeurIPS.* arXiv:1810.03292.
- Abnar, S., & Zuiderveld, W. (2020). Quantifying attention flow in transformers. *Proceedings of the 58th Annual Meeting of the ACL.* arXiv:2005.00928.
- Chefer, H., Gur, S., & Wolf, L. (2021). Transformer interpretability beyond attention visualization. *CVPR.* arXiv:2012.09838.
- Aas, K., Jullum, M., & Løland, A. (2021). Explaining individual predictions when features are dependent: More accurate approximations to Shapley values. *Artificial Intelligence, 298,* 103502. arXiv:1903.10464.
- Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of neural network representations revisited. *ICML.* arXiv:1905.00414.
- Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viégas, F., & Sayres, R. (2018). Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (TCAV). *ICML.* arXiv:1711.11279.
- Hewitt, J., & Liang, P. (2019). Designing and interpreting probes with control tasks. *EMNLP.* arXiv:1909.03368.
- Bibal, A., Cardon, R., Alfter, D., Wilkens, R., Wang, X., François, T., & Watrin, P. (2022). Is attention explanation? An introduction to the debate. *Proceedings of the 60th Annual Meeting of the ACL.* ACL Anthology 2022.acl-long.269.
- Krishna, S., Han, T., Gu, A., Pombra, J., Jabbari, S., Wu, S., & Lakkaraju, H. (2022). *The disagreement problem in explainable machine learning: A practitioner's perspective.* arXiv:2202.01602. **[preprint]**
- Nauta, M., Trienes, J., Pathak, S., Nguyen, E., Peters, M., Schmitt, Y., … Seifert, C. (2023). From anecdotal evidence to quantitative evaluation methods: A systematic review on evaluating explainable AI. *ACM Computing Surveys, 55*(13s), 1–42. https://doi.org/10.1145/3583558
- Hedström, A., Weber, L., Krakowczyk, D., Bareeva, D., Motzkus, F., Samek, W., Lapuschkin, S., & Höhne, M. M.-C. (2023). Quantus: An explainable AI toolkit for responsible evaluation of neural network explanations and beyond. *Journal of Machine Learning Research, 24*(34), 1–11. arXiv:2202.06861.
- Mamalakis, A., Ebert-Uphoff, I., & Barnes, E. A. (2022). *Investigating the fidelity of explainable AI methods for applications of convolutional neural networks in geoscience.* arXiv:2202.03407.
- Jönsson, P., & Eklundh, L. (2004). TIMESAT — a program for analyzing time-series of satellite sensor data. *Computers & Geosciences, 30*(8), 833–845. https://doi.org/10.1016/j.cageo.2004.05.006

### Phase 3 — LLM reporting
- Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W., Koh, P. W., Iyyer, M., Zettlemoyer, L., & Hajishirzi, H. (2023). FactScore: Fine-grained atomic evaluation of factual precision in long-form text generation. *EMNLP.* arXiv:2305.14251.
- Gao, T., Yen, H., Yu, J., & Chen, D. (2023). Enabling large language models to generate text with citations. *EMNLP.* arXiv:2305.14627.
- Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., & Zhu, C. (2023). G-Eval: NLG evaluation using GPT-4 with better human alignment. *EMNLP.* arXiv:2303.16634.
- Tanida, T., Müller-Sarnowski, F., Wachinger, C., & Rueckert, D. (2023). Interactive and explainable region-guided radiology report generation. *CVPR.*
- Singhal, K., Azizi, S., Tu, T., Mahdavi, S. S., Wei, J., Chung, H. W., … Natarajan, V. (2023). Large language models encode clinical knowledge. *Nature, 620,* 172–180.
- Kasner, Z., & Dušek, O. (2024). Beyond traditional benchmarks: Analyzing behaviors of open LLMs on data-to-text generation. *Proceedings of the 62nd Annual Meeting of the ACL.*
- Sclar, M., Choi, Y., Tsvetkov, Y., & Suhr, A. (2024). Quantifying language models' sensitivity to spurious features in prompt design. *ICLR.* arXiv:2310.11324.

---

# 6. Verification checklist (do before submitting the chapter)

All three research agents flagged that they had **no live web access** during this session. Before any citation lands in the submitted methods chapter, **verify on Scholar / arXiv / DOI**:

1. **EuroCropsML (Reuss et al., 2025)** — exact K-grid values and seed count from §Methods.
2. **Lisaius et al. (2026)** — exact protocol details (RF vs MLP primary, split sizes, K-grid).
3. **Forgaard et al. (2026) — THOR** — confirm authors and arXiv ID 2601.16011.
4. **TerraTorch (Gomes et al., 2024)** — confirm arXiv ID 2503.20669 + full author list + canonical citation form (paper vs technical report vs GitHub citation file).
5. **TESSERA encoder-weights distribution status** — confirm whether the encoder is public (HF Hub / GitHub) or only pre-computed embeddings are distributed. Affects whether LoRA on TESSERA is feasible.
6. **THOR TerraTorch wrapper availability** — confirm whether a TerraTorch backbone wrapper exists; if not, document the manual ingestion path.
4. **Krishna et al. (2022) — Disagreement Problem** — currently preprint; check whether a peer-reviewed venue has appeared (NeurIPS XAI workshop versions exist).
5. **Schulhoff et al. (2024) — Prompt Report** — preprint; check for venue.
6. **Biderman et al. (2024) — Reproducible LM Eval** — preprint.
7. **Yuan et al. (2024) — ChatEarthNet** — preprint; check venue.
8. **Zhang et al. (2024) — FinReport** — preprint; check venue.
9. **Nyborg et al. (2022) — TimeMatch** — confirm exact venue (CVPR-W vs RSE).
10. **Zhao et al. (2023) — composite vs time-series crop classification** — agent flagged title uncertainty; do targeted Scopus search.
11. **AlphaEarth annual-mismatch precedent paper** — agent referenced *"Harvesting AlphaEarth"* (arXiv:2601.00857); verify exists.
12. **Es et al. (2024) — RAGAS** — confirm EACL 2024 demo track citation.
13. All arXiv IDs above — sanity-check before pasting; agents flagged they did not fabricate but could not verify.

Also recommended: open the three `.docx` files under `Proposal/Proposal/GFM + XAI/` and deduplicate against § 2 of this document — the Phase 2 agent could not read those files in this session.
