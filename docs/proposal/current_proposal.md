# Geospatial Foundation Models for Transparent Agricultural Monitoring under Label-Scarce Conditions

## Candidate

- **Full name:** David Reyes
- **Student number:** 3598535
- **ITC E-mail address:** d.a.reyesmunoz@student.utwente.nl
- **Course:** M-GEO, 2026-2027, GEO-AI
- **Theme:** GFMs

## Supervisors

- **First supervisor:** Dr. Mahdi Farnaghi
- **Second supervisor:** Dr. Mariana Belgiu

---

# Description of the Proposed Research

## 1. Introduction

### Importance

Food systems are under increasing pressure from climate volatility and a growing global population. Earth observation (EO) has become the most practical means of tracking what is grown, where, and how it is performing at planetary scale, and open programmes such as Copernicus and Landsat (ESA, 2025; Wulder et al., 2022) have placed hundreds of petabytes of imagery in the public domain. Labels are more constrained: the annotated examples required to train supervised models remain expensive to produce, and each new monitoring task has so far demanded its own dataset and its own bespoke model. The bottleneck in operational EO has therefore shifted from data volume to annotation.

Because each task bears its own labelling cost, only the best-funded use cases can absorb it, such as continental cropland mapping, commercial farms and government land-use statistics. The result is a sharply skewed distribution of EO demand, in which a small head of applications attracts most of the field's investment. The long tail of smaller-scale needs, including regional crop monitoring, cooperative-level agronomic advisory and insurance loss assessment by value chain, remains largely unaddressed, because no single budget is large enough to justify a bespoke pipeline for each.

Agriculture spans this distribution. It is well served at the continental and national-statistics level, yet largely unserved at the scales where farming decisions are taken, namely the regional, cooperative and smallholder scales. At these scales, timely information on what is grown, where, and how it is performing underpins food-security monitoring, yield forecasting, drought and pest response, subsidy allocation and insurance settlement (FAO, 2023; Nakalembe & Kerner, 2023). The actors who most need this information, however, rarely work in remote sensing laboratories. They are farmers and cooperative agronomists, often operating in regions where label collection is costly and technical capacity is limited. Serving them requires more than accurate maps: the outputs must be interpretable, carry an explicit measure of uncertainty, and be communicated in natural language. Interpretability allows a user to see why a parcel was classified as it was and to weigh that against local agronomic knowledge. Uncertainty indicates whether a prediction is reliable enough to act upon. Natural-language reporting combines the two into a form the user can read directly, rather than as confusion matrices and probability tables. Addressing the long tail of agricultural EO demand is therefore not only a matter of cheaper models, but of cheaper models that are interpretable, uncertainty-aware and readable enough for non-specialist users to act upon.

### Background and related work

Geospatial Foundation Models (GFMs) directly address the label-scarcity side of this asymmetry. They are large neural networks pre-trained once on the abundant unlabelled archive and then reused for downstream tasks with only a small number of labels. GFMs learn general patterns of land-surface behaviour, including spectral signatures, their evolution through time, and the way spatial context constrains plausible classes. The result is a compact embedding that encodes much of this knowledge, so that adapting the model to a new question requires only a brief fine-tuning step or a simple classifier trained on the embeddings. GFMs therefore change the economics of EO in two ways: they make the head of the distribution more efficient, and they open the tail, since a new application now requires a few labelled examples rather than thousands (Mai et al., 2023).

Although GFMs are general purpose, this research focuses on agriculture, where satellite remote sensing already supports a wide range of monitoring tasks, including crop type, field health, irrigation, yield, deforestation, pest outbreaks and smallholder field boundaries (Weiss et al., 2020), and where label scarcity is most acute, particularly in smallholder and food-security contexts (Nakalembe & Kerner, 2023). The thesis adopts crop type classification as its concrete task, since it bears directly on food security, yield forecasting and policy reporting (FAO, 2023), and provides a natural setting in which to test how GFMs behave under label scarcity.

Three components are required before a GFM-based classifier can reach a non-specialist user:

1. **Interpretability.** GFM predictions derive from high-dimensional embeddings whose meaning is not self-evident. Explainable AI (XAI) methods link a prediction to specific input bands, dates or patches, so that the output can be checked against agronomic knowledge. SHAP, integrated gradients and attention attribution are the standard families of method.
2. **Uncertainty quantification.** A classification is actionable only when its confidence is known. Methods such as probability calibration, conformal prediction and ensemble disagreement attach a reliability estimate to each prediction, so that a confident prediction can be distinguished from a borderline one before it informs a decision.
3. **Reporting.** Even an explained and uncertainty-aware classification map is of limited value if its interpretation requires confusion matrices and probability tables. Large language models (LLMs) can convert classifications, their explanations and their uncertainties into structured agricultural reports in natural language, traceable to the underlying evidence.

These three requirements, together with the models and benchmarks on which they depend, structure the remainder of this section. Each building block is reviewed in turn. The discussion begins with the GFMs that produce the embeddings and the benchmarks that rank them, proceeds to the XAI and uncertainty methods that would render those embeddings transparent, and concludes with the LLMs that would convert their outputs into readable reports. In each case the review identifies where the building block currently falls short, and these shortfalls are formalised as the three knowledge gaps presented at the end of the section.

#### Geospatial foundation models

The field of GFMs has developed rapidly. A first generation appeared in 2022 and 2023, demonstrating that a single model pre-trained on large volumes of unlabelled satellite imagery could be reused across many downstream tasks. SatMAE (Cong et al., 2022) was among the earliest, a masked autoencoder adapted to multi-spectral and temporal Sentinel-2 imagery. Scale-MAE (Reed et al., 2023) added scale awareness, enabling a single encoder to handle imagery at different ground sampling distances. Prithvi-EO-1.0 (Jakubik et al., 2023), developed by NASA and IBM, was the first widely adopted open GFM backed by an operational space agency, pre-trained on Harmonized Landsat Sentinel-2 imagery. Presto (Tseng et al., 2023) adopted a different strategy, providing a lightweight transformer for pixel-level time series intended for compute-limited, temporally driven settings.

A second generation, released in 2024 and 2025, scaled these ideas further. Prithvi-EO-2.0 (Szwarcman et al., 2024) extended the family to between 300 and 600 million parameters with explicit temporal and geographic embeddings, improving on its predecessor by 8 % on GEO-Bench (Lacoste et al., 2023). AlphaEarth Foundations (Brown et al., 2025) fused optical, SAR, LiDAR, climate and text data into 64-byte pixel embeddings and reduced error by 23.9 % across fifteen benchmark tasks. TESSERA (Feng et al., 2025), which targets temporal embeddings of surface spectra, was independently evaluated by Lisaius et al. (2026) for crop classification in Senegal, where it outperformed competing approaches by 28 % in temporal transfer.

Two recent models extend this line of work. TerraMind (Jakubik et al., 2025), developed by IBM and ESA Φ-lab, is the first any-to-any generative foundation model for EO. It is pre-trained on nine million spatiotemporal multimodal samples (optical, SAR, elevation, land cover and others) and introduces an inference strategy termed Thinking-in-Modalities, in which the model synthesises a missing data type internally, for example generating a plausible SAR image from an optical input. THOR (Forgaard et al., 2026), developed by the Norwegian Computing Center for ESA's Foundation Models for Climate and Society programme, is the first compute-adaptive GFM, processing Sentinel-1, Sentinel-2 and Sentinel-3 imagery natively at resolutions from 10 to 1000 m.

These models have begun to be evaluated directly on agricultural tasks. Chang et al. (2024) evaluate foundation model embeddings for crop type mapping across five datasets spanning five continents, and find that Sentinel-2 specialised encoders transfer across regions more reliably than general-purpose alternatives, with roughly one hundred labelled images per setting sufficient for high overall accuracy. Ma et al. (2025) benchmark AlphaEarth on three agricultural tasks in the United States and report competitive accuracy together with a limited temporal sensitivity in the annual embeddings that constrains monitoring within a single growing season.

#### Benchmarks

Benchmarks are standardised collections of datasets, splits and evaluation protocols that allow different models to be compared on the same task under identical conditions. Early GFM benchmarks were narrow and biased towards North America and Europe. PANGAEA (Marsocci et al., 2024) addresses this with a standardised, geographically inclusive protocol spanning multiple resolutions, sensors and temporalities. Its defining feature is a limited-labels setting at 10 %, in which models are fine-tuned on only 10 % of the training labels, drawn by stratified sampling to preserve rare classes. Rankings shift substantially between the full-data and 10 % settings, and GFMs do not yet consistently outperform supervised baselines, which makes the protocol a valuable stress test for label-scarce settings. THOR (Forgaard et al., 2026) reports state-of-the-art results on it. GEO-Bench (Lacoste et al., 2023) provides a complementary suite for conventional fine-tuning and served as the official benchmark for Prithvi-EO-2.0.

Neither benchmark focuses on crop classification. The PANGAEA limited-labels evaluation combines several EO domains, and GEO-Bench is framed primarily around conventional fine-tuning with each task's full training set rather than around few-shot label budgets. Neither incorporates a cross-region transfer protocol, in which a model trained in one country is evaluated in another. How the latest generation of GFMs ranks on crop classification when only few labels per class are available, and whether that ranking holds when the training and evaluation regions differ, therefore remains untested under a shared, agriculture-focused protocol.

#### Explainability and uncertainty for GFMs

As noted at the start of this section, interpretability and uncertainty quantification are the layers that give a machine-learning output the transparency a non-specialist user requires in order to trust it. Both are now mature subfields of remote sensing. On the interpretability side, SHAP and permutation importance have been used to rank spectral bands and textural features in land-cover and urban-vegetation mapping, and Grad-CAM and its variants are routinely applied to localise class-relevant regions in scene classification; recent reviews catalogue dozens of such method and task pairings across EO applications (Höhl et al., 2024). On the uncertainty side, post-hoc calibration, ensemble disagreement and conformal prediction are increasingly used to attach per-pixel or per-parcel confidence to land-cover, change-detection and environmental-monitoring products, and systematic reviews now treat uncertainty quantification as a prerequisite rather than an addition for trustworthy spatio-temporal EO (Ferchichi et al., 2025). For GFM embeddings the situation is reversed: published XAI work on GFMs remains rare and typically considers one model at a time, and no study has examined a GFM-based crop classifier through XAI and uncertainty quantification jointly, nor compared its explanations and confidence against a raw-feature baseline or against agronomic knowledge.

#### Large language models in Earth observation

EO outputs are predominantly numerical and spatial: pixel-level classifications, confusion matrices, probability surfaces and time-series plots. The users who act on them, however, work primarily in natural language. LLMs provide a bridge between the two, converting model outputs into reports that a non-specialist can read. Two approaches dominate. The first trains a vision-language model directly on satellite imagery to answer questions about a scene; GeoChat (Kuckreja et al., 2024) is a representative example. The second provides an LLM with a set of external EO tools and allows it to decide when to call each; REMSA (Chen et al., 2025) follows this design, assisting users in selecting a GFM. Both lower the barrier to expertise, but neither produces a traceable agricultural report. A model trained jointly on imagery is difficult to audit, because its reasoning is embedded in the same network that interprets the pixels. A tool-calling model is easier to follow but is prone to hallucination when its domain context is limited.

A third option keeps the LLM away from the imagery altogether and provides it with a structured summary of the output of a separate classification model, together with the explanations behind those predictions. The LLM then composes the report from the summary rather than by interpreting the pixels directly. This approach has not yet been combined with a GFM and XAI in an agricultural setting, which is one of the gaps this thesis addresses.

### Research problem and knowledge gap

The literature reveals three gaps that, taken together, define the research problem addressed by this thesis.

- **Gap 1: No independent benchmark places the current generation of GFMs side by side under low-label, cross-region agricultural conditions.** The field is evolving rapidly: Prithvi-EO-2.0, AlphaEarth, TerraMind, TESSERA and THOR have all been released within roughly a year, each with different design choices and trade-offs. Neither PANGAEA nor GEO-Bench targets crop classification specifically or incorporates a cross-region transfer protocol, and the recent GFMs reviewed above have not been jointly evaluated under a shared agricultural setting.
- **Gap 2: GFM-based crop classifiers have not been examined jointly through XAI and uncertainty-quantification methods, nor compared with raw-feature baselines and agronomic knowledge.** GFMs are opaque encoders that compress satellite time series into high-dimensional embeddings whose internal logic is hidden from the user. Standard explanation methods such as SHAP have been applied to remote sensing more broadly, but none has been applied to GFM embeddings to determine whether what the model has learned aligns with agronomic knowledge such as phenology, crop calendars, texture or sensitivity to specific spectral bands.
- **Gap 3: No published system connects a GFM-based crop classifier, an explicit XAI layer and an LLM that uses those explanations as structured context for a traceable agricultural report.** In the absence of this connection, LLM-generated agricultural reports remain ungrounded and risk misrepresenting model behaviours they cannot inspect.

---

## 2. Objectives and Research Questions

**Main objective:**

To investigate the potential of geospatial foundation models as a transparent and scalable foundation for agricultural monitoring under label-scarce conditions.

**Sub-objectives and research questions**

Each sub-objective states a knowledge outcome the thesis aims to establish, paired with the research questions it answers.

- **SO1.** Produce a label-budget characterisation of GFM-based classifiers and a raw-feature baseline for crop classification, covering both in-country performance and cross-country transfer.
  - **RQ1.** How few labelled samples per class are required for GFM-based crop classifiers to match a raw-feature baseline?
- **SO2.** Evaluate the transparency and reliability of GFM-based crop classifications through explainability and uncertainty quantification.
  - **RQ2.1.** Can explainability methods identify the spectral and temporal information used by GFM-based crop classifiers?
  - **RQ2.2.** Can uncertainty quantification methods deliver reliable, calibrated confidence estimates for GFM-based crop predictions?
  - **RQ2.3.** Do the resulting explanations and confidence estimates agree with those obtained from the raw-feature baseline, and are they consistent with agronomic knowledge?
- **SO3.** Develop and evaluate an XAI-grounded LLM reporting pipeline that consumes GFM classifications and their explanations to generate agricultural reports traceable to the underlying evidence.
  - **RQ3.** Can XAI-derived information serve as effective structured context for an LLM that generates traceable, agronomically grounded agricultural monitoring reports?

---

## 5. Research Methods

### Study Area

The study area covers Estonia, Latvia and Portugal. Estonia and Latvia are Baltic countries in northern Europe, sharing a boreal-continental climate and large commercial cereal farms producing wheat, barley, rapeseed and grassland. Portugal lies at the south-western edge of the continent, with a Mediterranean climate and a more fragmented agricultural landscape that includes smallholder farms, vineyards, olive groves, and rotational maize and cereals.

The three countries differ in climate, parcel size and crop calendar, which makes them well suited to testing how a GFM-based classifier generalises across zones and is the contrast exploited by the cross-country transfer experiment in Phase 1.

Once the pipeline has been established on this primary study area, it is applied to the broader geography of the secondary study area, which extends the analysis to smallholder and tropical agricultural systems across Sub-Saharan Africa, South America, Central Asia, and additional temperate and tropical zones in Europe and North America. The climates, parcel sizes and crop calendars represented there differ markedly from those of the three primary countries and constitute the more demanding conditions that the secondary application is intended to probe.

### Datasets

Each of the two study areas is covered by one benchmark dataset. The primary dataset is **EuroCropsML** (Reuss et al., 2025), a public benchmark of approximately 706,000 labelled parcels across Estonia, Latvia and Portugal, each accompanied by a Sentinel-2 reflectance time series for 2021. Its labels are official parcel-level declarations harmonised across the three countries, which removes the cost of label collection. It provides a built-in K-shot evaluation protocol and supports cross-country transfer, which together make it well suited to the label-scarcity question of RQ1. The dataset is freely available on Zenodo.

The secondary dataset exercises the same pipeline under more demanding conditions, including smallholder and tropical agriculture beyond the temperate European setting of EuroCropsML. **CropHarvest** (Tseng et al., 2021) is a suitable candidate: it harmonises around twenty source datasets and spans Rwanda, Kenya, Ethiopia, Sudan, Mali, Tanzania, Uganda, Zimbabwe, Togo, France (including the tropical overseas departments of Réunion and Martinique), Germany, Canada, Brazil, Uzbekistan and Tajikistan, with polygon-based labels for the parcel-level subset used in this thesis. As with the GFM candidate set, this choice may be revised during the project should a more suitable alternative emerge.

### Selection of GFMs

Four pretrained Geospatial Foundation Models are used. TerraMind (Jakubik et al., 2025) and THOR (Forgaard et al., 2026) are open-weight models accessed through TerraTorch. AlphaEarth Foundations (Brown et al., 2025) is distributed as precomputed annual embeddings on the Earth Engine collection `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`. TESSERA (Feng et al., 2025) is distributed as precomputed pixel-time-series embeddings accessible through its Python sampling library. All four are openly accessible at the time of writing, although the candidate set may change during the project as new GFMs are released.

A practical distinction cuts across these models in how they are distributed. Some are released as model weights, so that the encoder can be loaded and run on new imagery, whereas others are released only as precomputed embeddings sampled from a fixed archive, with the encoder withheld. The difference is not merely logistical, since it sets the ceiling on explainability: attributing a prediction back to specific bands, dates or patches requires running gradients or perturbations through the encoder and is therefore possible only when the weights are available, whereas with embeddings alone the analysis is confined to the output vector. Phase 2 returns to this distinction as the embedding and backbone tiers of the explainability analysis.

### Development Phases

The research has three connected phases, each tied to a sub-objective and its research questions. Phase 1 develops a label-efficient benchmarking pipeline for crop classification with GFMs, delivering the label-budget characterisation of SO1 and answering RQ1. Phase 2 applies XAI to recover the spectral and temporal cues those models rely on (RQ2.1) and uncertainty-quantification methods to attach a calibrated confidence to each prediction (RQ2.2), then sets both against the raw-feature baseline and agronomic knowledge (RQ2.3), together covering SO2. Phase 3 converts those outputs into structured context that an LLM uses to write traceable agricultural reports, addressing SO3 and RQ3. The phases are first developed on the primary dataset and then re-run on the secondary dataset, under the more demanding conditions it presents.

#### Phase 1: Label-efficient benchmarking pipeline of GFMs for crop classification

Phase 1 addresses SO1 and RQ1. The first step is to construct a GFM benchmarking pipeline, which relies, among other tools, on TorchGeo (Stewart et al., 2022) for foundation-model wrappers and standard data modules for Sentinel-2 and HLS imagery. For GFMs released as model weights, namely TerraMind and THOR, TerraTorch handles backbone loading and the freeze-and-head configuration directly. For GFMs released as precomputed embeddings, namely AlphaEarth and TESSERA, the pipeline samples one vector per parcel, either at the centroid or as the zonal mean of the pixel embeddings within the polygon. These GFMs represent the most recent models available at the time of writing, although the final selection may change for reasons of feasibility and workflow.

EuroCropsML (Reuss et al., 2025) provides per-parcel Sentinel-2 reflectance time series, given as the spatial median over each parcel's pixels at every cloud-free acquisition during 2021. The raw baseline reduces this time series to a fixed-length feature vector in one of two ways. The first uses TIMESAT (Jönsson & Eklundh, 2004) to extract agronomic phenometric features: the NDVI and EVI peak value and day-of-year, the length of season, the sowing and harvest day-of-year, and the per-band mean and standard deviation. The second resamples the time series onto a fixed monthly grid and flattens it. One or both are used, depending on the stage of development.

Each GFM is evaluated with its encoder weights frozen, by training a series of lightweight to medium-capacity heads on the resulting embedding. These heads range from distance-based classifiers to shallow neural and tree-based models, which tests whether the answer to RQ1 depends on the choice of downstream classifier. Full fine-tuning is excluded, since it is not available for the precomputed-embedding models and tends to underperform shallow heads at small label budgets (Marsocci et al., 2024).

For both pipelines, the class set is restricted to the 15 to 20 most frequent crops per country, so that the metrics remain well defined when only a few labels are available. K denotes the percentage of the available training labels used per crop class, and it is varied from 1 % up to the full training set over the grid `{1, 5, 10, 20, 50, 100}` %, tracing a learning curve of accuracy against label budget; each value of K is repeated with at least five random draws so that the estimate at each point is stable. For every value of K, the trained classifier is evaluated on the same fixed held-out test set, so that any change in score reflects the training budget alone rather than a change in the evaluation data, which makes the curves comparable across budgets and across models. Macro-F1 with 95 % bootstrap confidence intervals is the headline metric, and spatial block cross-validation (Roberts et al., 2017) controls for the spatial autocorrelation that random splits would otherwise conceal. This in-region protocol is run independently within each of the three countries, Estonia, Latvia and Portugal, so that the label-budget characterisation is produced for every country rather than for a single one. Once the in-region results are stable, the same protocol is repeated as a cross-country transfer experiment between the three countries, on the intersection of each pair's top classes. The overall flow of Phase 1 is shown in Figure 1.

**Figure 1.** High-level flow of Phase 1.

#### Phase 2: Explainability and uncertainty analysis of GFM predictions

This phase asks which spectral and temporal cues each GFM has learned to exploit for crop classification (RQ2.1), whether those cues align with the raw-feature baseline and with agronomic knowledge (RQ2.3), and how reliably each model's confidence reflects its accuracy at the parcel level (RQ2.2).

Here, agronomic knowledge means the prior knowledge available in the literature on which characteristics matter most for each crop, such as phenology, crop calendars, texture, or sensitivity to specific spectral bands. It is the reference against which the explanations are checked, by asking whether the features the models rely on match what is already known to be important for the crop in question.

The explainability analysis is organised in two tiers that reflect how the different GFMs are distributed:

1. **Embedding tier.** Examines the GFM's frozen output vector. It applies to every model in scope, including the precomputed-embedding ones, AlphaEarth and TESSERA, where the backbone is not available.
2. **Backbone tier.** Examines the encoder itself, attributing predictions back to the raw imagery input (bands, dates, patches). It applies only to GFMs whose weights are released, namely Prithvi-EO-2.0, TerraMind and THOR, and not to AlphaEarth.

Both tiers are run wherever feasible, but the thesis leans more heavily on the backbone tier, because access to the full encoder-decoder architecture gives substantially more methodological flexibility than working with a fixed classification head over frozen embeddings.

##### Embedding tier

At the embedding tier, the aim is to identify which dimensions of the frozen GFM output drive each crop prediction (RQ2.1) and then to bind those dimensions to interpretable agronomic features. SHAP and permutation importance rank the dimensions by their per-class contribution on the head trained over the embeddings, and a progressive per-dimension ablation, following the protocol of Benavides-Martinez et al. (2026) on AlphaEarth, identifies the smallest subset that recovers baseline accuracy. The leading dimensions are not interpretable in themselves, so linear probes are trained from each of them onto known agronomic targets such as the NDVI and EVI peak day-of-year, the length of season, soil-moisture proxies, and per-band statistics. The R² of each probe yields a binding table that links GFM dimensions to agronomic concepts, advancing the interpretation from which dimensions matter to what those dimensions encode. The same SHAP and permutation-importance pipeline is applied to the raw-feature baseline head; because its inputs, the phenometric statistics or monthly composites, already lie in agronomic space, the GFM dimension importances after binding and the baseline feature importances fall on the same axes and can be compared directly, which addresses RQ2.3.

##### Backbone tier

At the backbone tier, gradient-, perturbation- and attention-based methods attribute predictions back to the raw imagery input (RQ2.1). The precise set of methods depends on each backbone's architecture (CNN versus ViT, single-frame versus temporal), but the core toolbox comprises Integrated Gradients for per-band and per-date attribution, Occlusion as a causal check through the systematic blanking of bands, timesteps or patches, and attention-based methods such as AttnLRP for ViT backbones. The choice of methods is not fixed in advance; faithfulness is evaluated for each method on each backbone, so that methods producing unreliable signal on a given model are flagged. The raw-feature baseline has no backbone through which to attribute, but SHAP and permutation importance on its phenometric or monthly inputs operate directly in agronomic terms and provide the axis against which the GFM band-and-date attributions are benchmarked, which addresses RQ2.3.

##### Uncertainty quantification

Each Phase 1 classification head is wrapped in two uncertainty-quantification procedures (RQ2.2), although further procedures may be added. Both are applied at the head level on top of the frozen GFM encoder, so that the resulting signal reflects how confident the classifier is in each parcel-level prediction.

**Monte Carlo Dropout** activates dropout layers in the head at inference and draws N stochastic forward passes for the same input. The variation across passes yields a predictive distribution that approximates the model's uncertainty about each parcel. It stands as the primary method for now, because it requires only a single trained head, adds negligible cost at inference, and fits the lightweight, frozen-embedding design of the pipeline.

**Deep Ensembles** train N classification heads on the same frozen embedding with different random seeds and read uncertainty from the spread of their predictions at inference. They typically yield stronger and better-calibrated estimates but are more costly, since N heads must be trained and stored, so they are retained as a heavier comparison against which the Monte Carlo Dropout estimates are checked.

Per-parcel uncertainty is reported as the predictive entropy over the N samples, and calibration is assessed using reliability diagrams and the Expected Calibration Error. These estimates accompany each prediction, feed back into the XAI comparison by indicating whether high-uncertainty parcels coincide with weak attributions, and feed forward into the Phase 3 context document. The overall flow of Phase 2 is shown in Figure 2.

**Figure 2.** High-level flow of Phase 2.

#### Phase 3: XAI-grounded LLM reporting

This phase addresses SO3 and RQ3. It generates a traceable agricultural report for a target zone by combining the Phase 1 predictions, the Phase 2 explanations and external context, then prompting an LLM to write the report.

The first step is the assembly of a **structured context document**. For a target zone, an assembly routine produces a JSON document organised into blocks so that the LLM can parse each section reliably. The document contains three blocks: (i) predictions; (ii) explanations; and (iii) external context, comprising weather variables from ERA5 reanalysis, geographic descriptors (country, elevation and soil class from an open soil database such as SoilGrids), and image-texture descriptors where available. The document is assembled per zone at inference time.

The second step is to **select an LLM and integrate it into the inference pipeline**. The model must follow the required output format reliably and must be reproducible, since otherwise the experiment cannot be repeated. It is then wrapped in a prompting routine that fixes its role, supplies worked examples of the report template, and supplies the context document for the target zone. Reports are evaluated on a sample of zones drawn from the Phase 1 held-out test partition.

---

## 6. Ethical Considerations, Risks and Contingencies

### Ethical considerations

This study uses published, anonymised benchmark datasets under appropriate licences and collects no new field data, so that it raises no human-subjects or personal-data concerns. Two issues nonetheless deserve attention.

The first concerns how the outputs are used. The pipeline is intended to inform decisions about crops, including subsidy allocation, insurance assessment and food-security response, and a confident but incorrect classification can push such decisions in the wrong direction. The risk is greater for non-specialists, who cannot inspect the model and may read a fluent report as authoritative. For this reason, uncertainty quantification, explanation and traceability are treated as core methodology.

The second concerns dual use. The same tooling that assists a cooperative agronomist also lowers the barrier to monitoring individual farmers without their knowledge, whether for fraud policing or commercial intelligence. Any operational use would remain subject to existing data-protection and agricultural-policy rules governing how parcel-level information may be processed.

### Risks and contingencies

*Data.* The labels in both datasets derive from administrative declarations rather than field survey, so that some parcels are mislabelled, which biases Macro-F1 and matters most at small label budgets. This is mitigated by restricting each country to its 15 to 20 most frequent, well-defined crops and by applying data-denoising strategies where necessary; because every model is evaluated on the same labels, the noise acts as a shared confounder that leaves the comparison between the GFMs and the baseline intact.

*Methodology.* Phase 3 assumes that Phase 2 yields explanations that are faithful and agronomically meaningful, which may not hold. If they prove weak, that is itself a reportable Phase 2 finding, and the Phase 3 context can still operate from predictions and calibrated uncertainty alone, with only those explanations that pass the faithfulness checks passed forward. LLM hallucination is contained by constraining the model to the structured context, generating at temperature zero, and scoring each report for traceability against its source.

*Scope and time.* The design is broad and could exceed the schedule, so the phases are prioritised: Phases 1 and 2 are core deliverables, whereas Phase 3 is highly valuable but the first to be reduced if necessary.

### Work Plan

The work is organised into six work packages (WP1 to WP6) over roughly ten months, from project start to thesis submission and defence. The milestones mark the completion of the three research questions. The table below lists the work packages, their activities, milestones and timing in project months.

| Work package | Description | Milestone | Duration |
|---|---|---|---|
| WP1: Setup | Set up the environment, integrate EuroCropsML, and implement the K-shot evaluation protocol with spatial block cross-validation and a battery of lightweight to medium-capacity heads. | | M1 |
| WP1: Setup | Load TerraMind and THOR through TerraTorch with frozen encoder weights, sample AlphaEarth and TESSERA per-parcel embeddings at the centroid or as polygon zonal means, and implement the raw baseline (TIMESAT phenometrics and the fixed monthly-grid flatten variant). | Pipeline working | M2 |
| WP2: Phase 1 | Run in-region experiments across Estonia, Latvia and Portugal with K ∈ {1, 5, 10, 20, 50, 100, 200} samples per class and at least five seeds per cell, reporting Macro-F1 with 95 % bootstrap confidence intervals. | | M3 |
| WP2: Phase 1 | Repeat the protocol as a cross-country transfer experiment between the three countries on the intersection of each pair's top classes, consolidate the learning curves, and begin writing Phase 1. | RQ1 answered | M4 |
| WP3: Phase 2 | Embedding tier: rank embedding dimensions with SHAP and permutation importance, run the AlphaEarth progressive per-dimension ablation (Benavides-Martinez et al., 2026), and train linear probes onto agronomic targets to build the binding tables. | | M5 |
| WP3: Phase 2 | Backbone tier: per-band and per-date attribution with Integrated Gradients, Occlusion and AttnLRP on ViT backbones, with faithfulness evaluation; head-wise Deep Ensembles and Monte Carlo Dropout with calibration via Expected Calibration Error and reliability diagrams. | RQ2.1, RQ2.2 and RQ2.3 answered | M6 |
| WP4: Secondary dataset | Restrict the secondary dataset to its polygon-labelled subset, sample embeddings at the centroid or as polygon zonal means, and re-run the Phase 1 protocol. | | M7 |
| WP4: Secondary dataset | Re-run the Phase 2 embedding-tier, backbone-tier and head-wise uncertainty procedures on the secondary dataset, and consolidate the results separately from the primary-dataset results. | Secondary-dataset results in hand | M8 |
| WP5: Phase 3 | Assemble the structured context document (predictions, explanations and external-context blocks), select and prompt the LLM, and generate reports on a sample of zones from the Phase 1 held-out test partition for evaluation. | RQ3 answered | M9 |
| WP6: Wrap-up | Final polish, defence preparation and submission. | Thesis submitted and defended | M10 |

---

# References

- Benavides-Martinez, I. F., et al. (2026). *What on Earth is AlphaEarth? Hierarchical structure and functional interpretability for global land cover.* arXiv. https://arxiv.org/abs/2603.16911
- Brown, C. F., Kazmierski, M. R., Pasquarella, V. J., Rucklidge, W. J., Samsikova, M., & Zhang, C. (2025). *AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data.* arXiv. https://arxiv.org/abs/2507.22291
- Chang, Y.-C., Stewart, A. J., Bastani, F., Wolters, P., Kannan, S., Huber, G. R., Wang, J., & Banerjee, A. (2024). *On the generalizability of foundation models for crop type mapping.* In IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2025). arXiv. https://arxiv.org/abs/2409.09451
- Chen, B., Bök, T. E., Rasti, B., Markl, V., & Demir, B. (2025). *REMSA: Foundation model selection for remote sensing via a constraint-aware agent.* arXiv. https://arxiv.org/abs/2511.17442
- Cong, Y., Khanna, S., Meng, C., Liu, P., Rozi, E., He, Y., Burke, M., Lobell, D. B., & Ermon, S. (2022). *SatMAE: Pre-training transformers for temporal and multi-spectral satellite imagery.* In Advances in Neural Information Processing Systems 35 (NeurIPS 2022). https://arxiv.org/abs/2207.08051
- European Space Agency. (2025). *Copernicus Data Space Ecosystem annual report 2024.* https://dataspace.copernicus.eu/news/2025-12-4-copernicus-data-space-ecosystem-cdse-releases-annual-report-2024
- Food and Agriculture Organization of the United Nations. (2023). *The state of food and agriculture 2023: Revealing the true cost of food.* FAO. https://doi.org/10.4060/cc7724en
- Feng, Z., Atzberger, C., Jaffer, S., Knezevic, J., Sormunen, S., Young, R., & Lisaius, M. C. (2025). *TESSERA: Temporal embeddings of surface spectra for Earth representation and analysis.* arXiv. https://arxiv.org/abs/2506.20380
- Ferchichi, A., Ferchichi, A., Hendaoui, F., Chihaoui, M., & Toujani, R. (2025). *Deep learning-based uncertainty quantification for spatio-temporal environmental remote sensing: A systematic literature review.* Neurocomputing, 639. https://doi.org/10.1016/j.neucom.2025.130242
- Forgaard, T., Reksten, J. H., Waldeland, A. U., Marsocci, V., Longépé, N., Kampffmeyer, M., & Salberg, A.-B. (2026). *THOR: A versatile foundation model for Earth observation climate and society applications.* arXiv. https://arxiv.org/abs/2601.16011
- Gomes, C., Blumenstiel, B., Almeida, J. L. de S., Oliveira, P. H. de, Fraccaro, P., Marti Escofet, F., Szwarcman, D., Simumba, N., Kienzler, R., & Zadrozny, B. (2025). *TerraTorch: The geospatial foundation models toolkit.* In IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2025). arXiv. https://arxiv.org/abs/2503.20563
- Höhl, A., Obadic, I., Fernández-Torres, M. Á., Najjar, H., Oliveira, D., Akata, Z., Dengel, A., & Zhu, X. X. (2024). *Opening the black-box: A systematic review on explainable AI in remote sensing.* IEEE Geoscience and Remote Sensing Magazine, 12(4), 261–304. https://doi.org/10.1109/MGRS.2024.3467001
- Jakubik, J., Roy, S., Phillips, C. E., Fraccaro, P., Godwin, D., & Zadrozny, B. (2023). *Foundation models for generalist geospatial artificial intelligence.* arXiv. https://arxiv.org/abs/2310.18660
- Jakubik, J., Yang, F., Blumenstiel, B., Scheurer, E., Sedona, R., & Maurogiovanni, S. (2025). *TerraMind: Large-scale generative multimodality for Earth observation.* In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). https://arxiv.org/abs/2504.11171
- Jönsson, P., & Eklundh, L. (2004). *TIMESAT—a program for analyzing time-series of satellite sensor data.* Computers & Geosciences, 30(8), 833–845. https://doi.org/10.1016/j.cageo.2004.05.006
- Kuckreja, K., Danish, M. S., Naseer, M., Das, A., Khan, S., & Khan, F. S. (2024). *GeoChat: Grounded large vision–language model for remote sensing.* In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). https://arxiv.org/abs/2311.15826
- Lacoste, A., Lehmann, N., Rodriguez, P., Sherwin, E., Kerner, H., & Lütjens, B. (2023). *GEO-Bench: Toward foundation models for Earth monitoring.* In Advances in Neural Information Processing Systems 36 (Datasets and Benchmarks Track). https://arxiv.org/abs/2306.03831
- Lisaius, M. C., Keshav, S., Blake, A., & Atzberger, C. (2026). *Embedding-based crop type classification in the Groundnut Basin of Senegal.* arXiv. https://arxiv.org/abs/2601.16900
- Lundberg, S. M., & Lee, S.-I. (2017). *A unified approach to interpreting model predictions.* In Advances in Neural Information Processing Systems 30 (pp. 4765–4774). https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
- Ma, Y., Shen, Y., Swatantran, A., & Lobell, D. B. (2025). *Harvesting AlphaEarth: Benchmarking the geospatial foundation model for agricultural downstream tasks.* arXiv. https://arxiv.org/abs/2601.00857
- Mai, G., Huang, W., Sun, J., Song, S., Mishra, D., & Liu, N. (2023). *On the opportunities and challenges of foundation models for geospatial artificial intelligence.* arXiv. https://arxiv.org/abs/2304.06798
- Marsocci, V., Jia, Y., Le Bellier, G., Kerekes, D., Zeng, L., & Hafner, S. (2024). *PANGAEA: A global and inclusive benchmark for geospatial foundation models.* arXiv. https://arxiv.org/abs/2412.04204
- Nakalembe, C., & Kerner, H. (2023). *Considerations for AI-EO for agriculture in Sub-Saharan Africa.* Environmental Research Letters, 18(4), Article 041002. https://doi.org/10.1088/1748-9326/acc476
- Reed, C. J., Gupta, R., Li, S., Brockman, S., Funk, C., & Clipp, B. (2023). *Scale-MAE: A scale-aware masked autoencoder for multiscale geospatial representation learning.* In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). https://arxiv.org/abs/2212.14532
- Reuss, J., Macdonald, J., Becker, S., Richter, L., & Körner, M. (2025). *The EuroCropsML time series benchmark dataset for few-shot crop type classification in Europe.* Scientific Data. https://www.nature.com/articles/s41597-025-04952-7
- Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F., & Dormann, C. F. (2017). *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure.* Ecography, 40(8), 913–929. https://doi.org/10.1111/ecog.02881
- Stewart, A. J., Robinson, C., Corley, I. A., Ortiz, A., Lavista Ferres, J. M., & Banerjee, A. (2022). *TorchGeo: Deep learning with geospatial data.* In Proceedings of the 30th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems. https://doi.org/10.1145/3557915.3560953
- Szwarcman, D., Roy, S., Fraccaro, P., Gomes, C., & Zadrozny, B. (2024). *Prithvi-EO-2.0: A versatile multi-temporal foundation model for Earth observation applications.* arXiv. https://arxiv.org/abs/2412.02732
- Tseng, G., Zvonkov, I., Nakalembe, C., & Kerner, H. (2021). *CropHarvest: A global satellite dataset for crop type classification.* In Advances in Neural Information Processing Systems 34 (Datasets and Benchmarks Track). https://openreview.net/forum?id=JtjzUXPEaCu
- Tseng, G., Cartuyvels, R., Zvonkov, I., Purohit, M., Rolnick, D., & Kerner, H. (2023). *Lightweight, pre-trained transformers for remote sensing time series.* In NeurIPS 2023 Workshop on Tackling Climate Change with Machine Learning. https://arxiv.org/abs/2304.14065
- Weiss, M., Jacob, F., & Duveiller, G. (2020). *Remote sensing for agricultural applications: A meta-review.* Remote Sensing of Environment, 236, Article 111402. https://doi.org/10.1016/j.rse.2019.111402
- Wulder, M. A., Roy, D. P., Radeloff, V. C., Loveland, T. R., Anderson, M. C., & Johnson, D. M. (2022). *Fifty years of Landsat science and impacts.* Remote Sensing of Environment, 280, Article 113195. https://doi.org/10.1016/j.rse.2022.113195

---

# Annexes

## Annex A: Data Management Plan

**Data.** The project uses only open agricultural and Earth-observation data; no personal data is collected, generated or processed at any stage. Secondary data includes EuroCropsML, CropHarvest, Sentinel-1 and Sentinel-2 imagery via Copernicus, ERA5 climate data, and the four GFMs in scope (TerraMind, THOR, AlphaEarth, TESSERA), each obtained under its own open licence. Primary data generated by the project consists of GFM embeddings, trained classification heads, experiment logs and metrics, XAI outputs, LLM-generated reports, code and figures.

**Organisation and documentation.** Code is versioned in a private Git repository. Large derived artefacts such as embedding caches and model checkpoints are tracked by URL or DVC pointers rather than committed directly. A top-level README documents the project structure, software and model versions, dataset sources and their licences, and reproduction instructions.

**Storage and access.** Working copies are kept on UTwente / ITC institutional storage (project drive), with a backup on UTwente cloud storage. Access during the project is limited to the candidate and the two supervisors. Because no personal or otherwise sensitive data is involved, no additional encryption is required.

**Archiving.** At the end of the thesis, code, experiment results, XAI outputs, figures, sample LLM reports, and trained classification heads (where the upstream model licence allows it) are deposited in the ITC MSc research-data archive for the prescribed seven-year retention period. Secondary data and precomputed embedding caches are not archived because they remain available from their original sources or can be regenerated from the archived code.