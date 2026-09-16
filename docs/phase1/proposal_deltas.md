# Departures from the proposal and from CLAUDE.md

Every point at which `docs/phase1/protocol.md` departs from `docs/proposal/current_proposal.md` or from `CLAUDE.md`, with the original wording, the replacement, and the reason. Use this list to correct those two documents. Line numbers refer to `docs/proposal/current_proposal.md` as of commit `db663cb`.

Twenty-four deltas. Deltas 1, 2, 3, 6, 13 and 21 are the consequential ones.

---

## Delta 1. The definition of K is contradictory and the percentage definition is wrong

**Severity: blocking.**

**Proposal, line 135:**
> "K denotes the percentage of the available training labels used per crop class, and it is varied from 1 % up to the full training set over the grid `{1, 5, 10, 20, 50, 100}` %, tracing a learning curve of accuracy against label budget"

**Proposal, line 208 (work plan, WP2):**
> "Run in-region experiments across Estonia, Latvia and Portugal with K ∈ {1, 5, 10, 20, 50, 100, 200} samples per class"

**CLAUDE.md, Phase 1 section:**
> "a label-budget grid K" ... and, under "Open inconsistency to resolve before WP2", the same contradiction is restated.

**Replacement, for line 135:**
> "K denotes the number of labelled samples per crop class, and it is varied over the grid `{1, 5, 10, 20, 50, 100, 200, 500}` samples per class together with the full training pool, tracing a learning curve of Macro-F1 against label budget"

**Replacement, for line 208:**
> "Run in-region experiments across Estonia, Latvia and Portugal with K ∈ {1, 5, 10, 20, 50, 100, 200, 500, pool} samples per class"

**Reason.** The two passages describe different experiments and produce different curves. The absolute-count definition is correct for three independent reasons. RQ1 asks "how few labelled samples per class", so the answer must be a count and not a proportion. K-shot conventionally means shots per class in the few-shot literature. A percentage confounds the label budget with country size, so one per cent of Latvia and one per cent of Portugal are budgets differing by a large factor, and the three national curves could not be placed on a common axis, which is precisely what the proposal asks for in the same sentence. The published EuroCropsML benchmark also uses absolute counts, so the percentage definition would additionally forfeit comparability. The grid is the union of the work plan's grid and the published grid `{1, 5, 10, 20, 100, 200, 500, all}`, which retains the point at fifty and adds the point at five hundred at negligible cost.

**Also correct:** `CLAUDE.md`, the "Open inconsistency to resolve before WP2" note should be replaced by a statement of the resolution rather than of the problem.

---

## Delta 2. The official EuroCropsML protocol is described as providing more than it does

**Severity: blocking.**

**Proposal, line 113:**
> "It provides a built-in K-shot evaluation protocol and supports cross-country transfer, which together make it well suited to the label-scarcity question of RQ1."

**CLAUDE.md, Datasets section:**
> "Built-in K-shot protocol and cross-country transfer protocol."

**Replacement, for line 113:**
> "It provides a built-in K-shot evaluation protocol for transnational transfer into Estonia, with Latvia, optionally together with Portugal, as the source. That protocol is adopted unchanged as a benchmark track, so that the results of this thesis can be placed beside the published ones. It does not provide in-country protocols for Latvia or Portugal, it applies no spatial control, and its single random seed governs the test partition and the support draw jointly, so the thesis extends it with a spatially blocked protocol covering all three countries in-country and nine directed transfer settings."

**Reason.** Inspection of the installed `eurocropsml` package, version 0.4.1, establishes that the four shipped split configurations all have Latvia, optionally with Portugal, as the pre-training regions and `EE` as the fine-tuning region. There is no configuration in which Latvia or Portugal is the target. The fine-tuning split is produced by two calls to `sklearn.model_selection.train_test_split` with no `stratify` argument and no spatial information, and the companion benchmarking paper states that the entire Estonian data is randomly split into train, validation and test. `EuroCropsSplit.random_seed` is a single integer controlling both the partition and the support draw, so several support draws on a fixed test partition cannot be obtained from the official artefacts. The proposal's sentence is therefore an overstatement that would mislead a reader into thinking the protocol could simply be adopted.

---

## Delta 3. Spatial block cross-validation is named but not specified, and it conflicts with the fixed test set as written

**Severity: blocking.**

**Proposal, line 135:**
> "Macro-F1 with 95 % bootstrap confidence intervals is the headline metric, and spatial block cross-validation (Roberts et al., 2017) controls for the spatial autocorrelation that random splits would otherwise conceal."

**CLAUDE.md, Phase 1 section:**
> "the same fixed held-out test set at every K; Macro-F1 with 95 % bootstrap confidence intervals; spatial block cross-validation (Roberts et al., 2017)"

**Replacement, for line 135:**
> "Macro-F1 with 95 % bootstrap confidence intervals is the headline metric. Spatial autocorrelation is controlled by partitioning whole blocks of a square tessellation of the ETRS89-LAEA plane, whose edge length is estimated per country from the label-agreement decay of the crop labels, with an exclusion buffer removing training parcels adjacent to any held-out block, whose width is set from the largest patch footprint in the sweep so that a test parcel's patch cannot contain a training parcel. The primary protocol holds a single spatial partition fixed at every budget, and block cross-validation over five spatial partitions is reported as a robustness check."

**Reason.** Two distinct problems. First, "spatial block cross-validation" names a family of methods without specifying the blocking unit, the block size, the fold assignment or the interaction with K-shot sampling, so it is not implementable as written. Second, a cross-validation and a fixed held-out test set are different designs, and the proposal asks for both in the same sentence without saying how they coexist. The protocol resolves this by making the fixed single partition primary, which is what makes the learning curves comparable, and making the five-partition cross-validation a robustness check on how far the conclusions depend on which blocks were held out. See `docs/phase1/protocol.md`, sections 3 and 3.6.

---

## Delta 4. The class set restriction needs a stated eligibility rule

**Proposal, line 135:**
> "For both pipelines, the class set is restricted to the 15 to 20 most frequent crops per country, so that the metrics remain well defined when only a few labels are available."

**Replacement:**
> "For both pipelines, the class set is restricted per country to the most frequent crops, subject to two eligibility conditions applied after partitioning: a class is retained only if its training pool holds at least five hundred parcels, so that every point of the budget grid is attainable, and only if the test partition holds at least two hundred parcels of it, so that per-class F1 is meaningful. Classes failing either condition are dropped and the drop is recorded in the split manifest."

**Reason.** "The 15 to 20 most frequent" is an ambiguous instruction that yields a different class set depending on where the cut is taken, and it makes no reference to whether the classes retained can actually support the largest budget or the per-class metric. The eligibility rule makes the number of classes an outcome of stated thresholds rather than a free choice. Provisional figures from the exploratory catalogue give ten classes for Estonia, fifteen for Latvia and six for Portugal, so the "15 to 20" range is in any case not attainable for two of the three countries.

---

## Delta 5. The test set composition is unspecified

**Proposal, line 135:**
> "For every value of K, the trained classifier is evaluated on the same fixed held-out test set"

**Addition after that sentence:**
> "The test set is a naturally distributed spatial sample of the parcel population rather than a class-balanced one. Macro-F1 already weights every class equally in the metric; balancing the test set as well would remove the precision penalty that a rare class incurs from false positives arriving from abundant classes, which is the operational failure mode of interest. Natural priors additionally allow overall accuracy and Cohen's kappa to be computed on the same predictions, which is what makes the thesis track and the benchmark track commensurable."

**Reason.** Whether the test set is balanced changes what Macro-F1 measures, and the proposal is silent on it. The argument is set out in `protocol.md`, section 4.2.

---

## Delta 6. Hyperparameter selection at small budgets is not addressed

**Severity: consequential.**

**Proposal, line 133:**
> "Each GFM is evaluated with its encoder weights frozen, by training a series of lightweight to medium-capacity heads on the resulting embedding."

**Addition after that sentence:**
> "Head hyperparameters cannot be tuned honestly on a five-label support set, and using the held-out validation partition to select per budget would inject thousands of labels of supervision into a cell nominally holding five. Each combination of feature set, head and country therefore receives a single hyperparameter configuration, selected once at the full budget on the spatially separate validation partition and frozen across every budget and every draw. The only exceptions are hyperparameters that must scale with the sample size, which follow rules declared in advance that read the sample size alone and never the labels. A support-internal cross-validation variant is reported at budgets of twenty and above as a sensitivity check, never as the headline."

**Reason.** The proposal does not mention validation or model selection for the heads at all. Left unstated, the natural implementation would select per budget on the validation partition, which would silently invalidate every small-budget result and therefore the answer to RQ1. The bias introduced by the chosen scheme runs against the low-budget end, which is the conservative direction for the thesis's own claim, and is stated explicitly rather than concealed. See `protocol.md`, section 4.5.

---

## Delta 7. The bootstrap is unspecified and the two variance components are conflated

**Proposal, line 135:**
> "each value of K is repeated with at least five random draws so that the estimate at each point is stable. ... Macro-F1 with 95 % bootstrap confidence intervals is the headline metric"

**Replacement of the confidence-interval clause:**
> "Macro-F1 is the headline metric. Two sources of variation are reported separately, because conflating them overstates precision. Variation across the support draws at a fixed budget is reported as the mean, the standard deviation and the range across draws. Variation attributable to the finite test sample is estimated by a block cluster bootstrap, resampling whole test blocks with replacement over one thousand replicates, since an independent parcel bootstrap on spatially clustered test data returns an interval that is too narrow. The headline interval pools the bootstrap replicates across draws, and the variance decomposition into a draw component and a test component is given numerically in every results table."

**Reason.** "95 % bootstrap confidence intervals" does not say what is resampled. Resampling parcels independently is the obvious default and it is wrong here, because parcels within a field block are strongly correlated. Reporting one interval that silently mixes the draw variance and the test variance is the more common error and it overstates precision, most severely at small K where the draw component dominates. See `protocol.md`, section 4.7.

---

## Delta 8. The number of draws is raised at small budgets and lowered at the full budget

**Proposal, line 135:**
> "each value of K is repeated with at least five random draws"

**Replacement:**
> "each value of K is repeated with ten random draws at K of one, five and ten, five draws at K from twenty to five hundred, and three at the full budget, where the support set is the whole training pool and the only remaining variation is the head's own initialisation."

**Reason.** Five draws everywhere spends the same effort on the cheapest and highest-variance cells as on the most expensive and lowest-variance ones. At K equal to one the between-draw variance is at its maximum and a single fit costs milliseconds; at the full budget the support set is the entire pool and repeating it five times measures only the head's initialisation, at a cost of minutes per fit. The proposal's floor of five is met or exceeded everywhere except at the full budget, where the departure is stated.

---

## Delta 9. Support sets must be nested and spread across blocks

**Not present in the proposal.** New specification.

**Addition to line 135:**
> "Support sets are nested across budgets, so that moving along the learning curve adds labels rather than exchanging them, and they are drawn spread across spatial blocks, so that K labels come from as many distinct field blocks as the class permits. Without the second condition, an unconstrained draw at K equal to twenty can return twenty parcels from a single field cluster, which is closer to one independent annotation than to twenty."

**Reason.** Neither property follows from "at least five random draws". Nesting removes a spurious source of variation between adjacent points of the curve. Block spreading is what makes the x-axis of the learning curve mean what it claims, given the strong spatial clustering of parcels by crop. The unconstrained variant is retained as a sensitivity check, and on the exploratory catalogue block spreading raises the mean number of distinct blocks per class at K equal to twenty to the maximum of twenty.

---

## Delta 10. Every model must see the identical support set

**Not present in the proposal.** New requirement.

**Addition to line 135:**
> "The support set at a given country, class, budget and draw seed is a deterministic function of the split configuration alone, and is independent of the model, the head and the order in which the sweep is executed, so that no comparison between models can be confounded by a difference in the labels each was given."

**Reason.** This is an obvious requirement that is easy to violate accidentally, for example by seeding a global generator once per run rather than deriving the draw from the split context. It is enforced in the reference implementation by deriving every generator from a hash of the context parts, and it is asserted by the test suite.

---

## Delta 11. The full budget is not the whole of a country's labels

**Proposal, line 135:**
> "it is varied from 1 % up to the full training set"

**Replacement, in the corrected absolute-count wording:**
> "together with the full training pool, which is capped at twenty thousand parcels per class so that the full-budget fits and the embedding extraction remain tractable; the cap is recorded and the full-budget point is labelled as the pool rather than as one hundred per cent of the country's labels."

**Reason.** Without a cap the full-budget point scales with the whole 706,683-parcel archive, which drives the embedding extraction cost, which is the true bottleneck of Phase 1, for no scientific gain: the learning curve has long since saturated. Labelling the point honestly matters, because "one hundred per cent" would be false.

---

## Delta 12. The matching criterion of RQ1 needs a statistical definition

**Not present in the proposal.** RQ1 is stated without a decision rule.

**Proposal, line 26 and elsewhere:**
> "RQ1. How few labelled samples per class are required for GFM-based crop classifiers to match a raw-feature baseline?"

**Addition to line 135:**
> "The matching budget is defined as follows. Because the foundation model and the baseline are evaluated on the same fixed test parcels, they are compared as a paired difference on each bootstrap replicate rather than as two independent intervals. The matching budget is the smallest K on the grid at which the 2.5th percentile of the paired difference between the foundation model at budget K and the raw-feature baseline at the full budget is no worse than a margin of 0.02 Macro-F1, and at which the same holds at every larger K on the grid. The matching budget at equal budget, and the budget at which the foundation model reaches ninety-five per cent of its own full-budget score, are reported alongside. Where no budget on the grid satisfies the criterion the result is reported as not attained, and no interpolation between grid points is performed."

**Reason.** RQ1 is the thesis's central question and, as written, "match" has no operational meaning. The obvious default, overlap of two confidence intervals, has no defined error rate and is conservative in a way that varies with the relative interval widths, and it discards the pairing that the shared test set makes available. The margin of 0.02 is declared in advance and must not be changed after results are seen. See `protocol.md`, section 4.8.

---

## Delta 13. The cross-country transfer design is under-specified in three ways

**Severity: consequential.**

**Proposal, line 135:**
> "Once the in-region results are stable, the same protocol is repeated as a cross-country transfer experiment between the three countries, on the intersection of each pair's top classes."

**Proposal, line 209 (work plan, WP2):**
> "Repeat the protocol as a cross-country transfer experiment between the three countries on the intersection of each pair's top classes"

**Replacement, for line 135:**
> "Once the in-region results are stable, the protocol is repeated as a cross-country transfer experiment over six ordered country pairs and three many-to-one settings, on the intersection of the evaluated class sets of the countries involved. With a frozen encoder only the head is fitted, so three regimes are distinguished and all three are run: zero-shot, in which the head is trained on the source and applied unchanged to the target; target-only at the same budget, which is the reference line; and source plus K target labels, which is the practical setting and the one the published benchmark evaluates. The head is retrained on the intersected label space rather than masked at inference. Within one transfer setting every reported number is computed on the identical label space and the identical target test subset."

**Reason.** Three gaps. First, "between the three countries" does not say whether transfer is directed; it is not symmetric, so six ordered pairs are required, and the many-to-one settings are worth running because the official benchmark's `latvia_portugal_vs_estonia` configuration exists precisely to ask whether adding a distant source helps. Second, the proposal does not say what transfers. With a frozen encoder the only thing that can transfer is the head, and zero-shot and few-shot answer different questions, so leaving it unstated would leave the experiment undefined. Third, whether the head is retrained on the intersection or masked at inference changes the result: masking leaves the decision boundaries shaped by classes that cannot occur, which conflates representation quality with an artefact of the label space.

---

## Delta 14. The Portugal transfer pairs have almost no shared label space

**Not present in the proposal.** New finding, with a specified remedy.

**Addition to line 135, after the transfer sentence:**
> "At full HCAT depth the top-class sets of Portugal and each Baltic country intersect in approximately one class, which is not a usable label space. Transfer settings involving Portugal are therefore run at the six-digit HCAT level, which raises the intersection to approximately three classes, and are reported as a distinct low-overlap regime whose label-space size is printed in every caption and whose scores are interpreted only against their own target-only reference on the same label space."

**Reason.** Measured on the thirty-thousand-parcel exploratory catalogue: at full depth the intersections are Estonia with Latvia nine classes, Estonia with Portugal one, Latvia with Portugal one; at six digits they are six, three and three. Without this correction, two of the six ordered pairs and two of the three many-to-one settings would be run on a label space of one class, which is not a classification problem. A Macro-F1 over three classes is also not comparable with a Macro-F1 over nine and must never be placed on the same axis without annotation. The figures must be recomputed once `configs/class_scheme_eurocropsml.yaml` exists.

---

## Delta 15. The class prior shift confound in transfer is not addressed

**Not present in the proposal.** New requirement.

**Addition to line 135:**
> "Portugal's class distribution and crop calendar differ from those of the Baltic countries strongly enough that a fall in transfer score may reflect a shift in the class prior rather than a failure of the representation. The two are separated by four instruments reported for every transfer setting: per-class F1 on the intersected label space together with its per-class difference from the target-only reference; a prior-corrected variant using the expectation-maximisation procedure of Saerens, Latinne and Decaestecker (2002); a prior-matched target test subsample; and the transfer gap plotted against the Jensen-Shannon divergence of the class priors and against a phenological distance derived from the baseline phenometrics."

**Reason.** Without this, a low Portugal transfer score would be reported as evidence that the foundation model's representation does not generalise, when it may be evidence only that Portugal grows different crops in different proportions. The distinction is the difference between a claim about representations and a claim about label distributions, and only the first is within the scope of SO1.

---

## Delta 16. The learning-curve metric is named inconsistently

**Proposal, line 135:**
> "tracing a learning curve of accuracy against label budget"

**Replacement:**
> "tracing a learning curve of Macro-F1 against label budget"

**Reason.** The same sentence names Macro-F1 as the headline metric three clauses later. "Accuracy" here is loose usage, but on a class distribution in which one class holds forty-five per cent of parcels the two metrics behave very differently, so the loose usage is not harmless. Overall accuracy is retained as a secondary metric, reported alongside Cohen's kappa, for comparability with the published benchmark.

---

## Delta 17. The head battery should be named

**Proposal, line 133:**
> "These heads range from distance-based classifiers to shallow neural and tree-based models, which tests whether the answer to RQ1 depends on the choice of downstream classifier."

**Replacement:**
> "Four heads are used, spanning the three families: a nearest-class-mean classifier and a k-nearest-neighbour classifier as the distance-based members, a multinomial logistic regression and a shallow multi-layer perceptron as the linear and neural members, and histogram gradient boosting as the tree-based member. In the transfer sweep the battery is reduced to one head per family, so the head-invariance claim is evidenced in-country and carried over to transfer."

**Reason.** The families are named but the members are not, so the experiment is not reproducible as written. The reduction in the transfer sweep is a deliberate compute decision, stated as a limitation in `protocol.md`, section 6.4.

---

## Delta 18. The phrase "one or both" for the baseline variants is not a protocol

**Proposal, line 131:**
> "The second resamples the time series onto a fixed monthly grid and flattens it. One or both are used, depending on the stage of development."

**Replacement:**
> "Both variants are carried through the in-country sweep. The stronger of the two, judged on the in-country full-budget Macro-F1, is carried alone into the transfer sweep, and the selection is reported."

**Reason.** "One or both, depending on the stage of development" leaves the identity of the baseline undetermined, and RQ1 is a question about matching the baseline, so the baseline must be fixed by a stated rule rather than by circumstance. Both are cheap in-country; carrying only the stronger into transfer is a compute decision that is stated.

---

## Delta 19. The TIMESAT dependency should be reduced to a specification

**Proposal, line 131:**
> "The first uses TIMESAT (Jönsson & Eklundh, 2004) to extract agronomic phenometric features"

**Replacement:**
> "The first extracts TIMESAT-style phenometric features (Jönsson and Eklundh, 2004), namely the NDVI and EVI peak value and day-of-year, the length of season, the sowing and harvest day-of-year, and the per-band mean and standard deviation, computed within the project pipeline from the per-parcel time series."

**Reason.** TIMESAT is a separate program with its own input format and licence, and routing 706,683 parcel time series through it would be a substantial engineering detour for a baseline. The features are the point, not the software. Naming the features and the reference makes the baseline reproducible and removes an external dependency from the critical path. The phrasing already used in `CLAUDE.md`, "TIMESAT-style phenometrics", is the correct one and should replace the proposal's wording.

---

## Delta 20. The feature cache and the result manifest are not specified

**Not present in the proposal beyond a general statement.**

**Proposal, Annex A:**
> "Large derived artefacts such as embedding caches and model checkpoints are tracked by URL or DVC pointers rather than committed directly."

**Addition to line 129:**
> "Features are extracted once per combination of model, input representation and sampling rule into a parcel-keyed cache, over a parcel set fixed in advance, and are reused across every budget, draw, head and experiment. A single encoder therefore contributes several cache entries, one per representation it consumes. Raw patches are transient and are not cached, since only the encoder output is reused; a small stratified sample is nonetheless retained for the Phase 2 backbone tier, which requires the input tensor in order to attribute to it. No budget, draw, head or experiment may trigger an extraction; a cache miss is an error. Every result file carries the label budget, the country, the split protocol, the draw seed, the model, the input representation and its comparison group, the head, the configuration hash and a hash of the features consumed."

**Reason.** AlphaEarth and TESSERA sampling is network bound, and patch extraction for the open-weight models is download bound; together they are the true bottleneck of Phase 1. Without an explicit rule forbidding on-demand extraction, a naive implementation would refetch embeddings per experiment cell, which would dominate the project's wall-clock time. The manifest requirement is the repository's own reproducibility convention, stated in `CLAUDE.md`, made concrete. See `protocol.md`, sections 6.2 and 6.3.

---

## Delta 21. Input representation is an uncontrolled factor that would confound the whole of SO1

**Severity: blocking.**

**Proposal, line 129:**
> "For GFMs released as model weights, namely TerraMind and THOR, TerraTorch handles backbone loading and the freeze-and-head configuration directly. For GFMs released as precomputed embeddings, namely AlphaEarth and TESSERA, the pipeline samples one vector per parcel, either at the centroid or as the zonal mean of the pixel embeddings within the polygon."

**Proposal, line 131:**
> "EuroCropsML (Reuss et al., 2025) provides per-parcel Sentinel-2 reflectance time series, given as the spatial median over each parcel's pixels at every cloud-free acquisition during 2021."

**Replacement, to follow line 129:**
> "The four models do not consume the same input. TerraMind and THOR consume multi-date Sentinel-2 patches, which raw imagery downloads make available; AlphaEarth and TESSERA consume a sampled embedding vector; the raw-feature baselines consume the per-parcel median time series. These inputs differ in spatial support, since a patch carries the parcel's neighbourhood, a parcel median or zonal mean carries the parcel interior, and a centroid sample carries one pixel. A difference in Macro-F1 between models consuming different inputs is therefore not attributable to the encoder. The input representation is consequently treated as a declared experimental factor with six levels, and three comparison groups are defined by spatial support. The parcel-support group, in which every model sees the parcel interior and nothing else, is the headline group for RQ1, because within it the encoder is the only factor that varies. Three bridge experiments, each holding the encoder fixed and varying only the representation, estimate the representation offset and are reported as a table alongside the main results."

**Reason.** This is the most serious methodological gap in the Phase 1 design as proposed. SO1 is entirely a comparison between encoders. If TerraMind is evaluated on a 2.24 km patch and AlphaEarth on a single centroid pixel, then any conclusion of the form "TerraMind is more label-efficient than AlphaEarth" is unsupported, because the patch carries field boundaries, parcel shape, texture and the crops of neighbouring parcels that the centroid pixel does not. The proposal does not notice the asymmetry, and the passage at line 129 in fact institutionalises it by describing a different input pipeline per model without remarking that this makes the models incomparable. The comparison groups prevent the error and the bridges quantify it. The likely outcome is itself reportable: if the encoder-free Bridge C accounts for most of the cross-group difference, the finding is that the input representation matters more than the choice of foundation model, which is a substantive contribution rather than a negative result. See `docs/phase1/protocol.md`, sections 2.4 and 2.5.

**Also correct:** `CLAUDE.md`, the "Selection of GFMs" table should gain a representation column, and the sentence "The weights-versus-embeddings split is not merely logistical: it sets the ceiling on explainability" should be extended, since that split also sets the input representation and therefore bears on Phase 1 and not only on Phase 2.

---

## Delta 22. The buffer width and the pool cap are consequences of the patch footprint

**Not present in the proposal.** New requirement, following from Delta 21.

**Addition to line 135:**
> "Because a patch-based model sees a fixed footprint around the parcel centroid, a test parcel's patch could otherwise contain a training parcel. The exclusion buffer is therefore set to at least half the diagonal of the largest patch footprint used by any model in the sweep, 1,600 m for a footprint of 224 pixels at 10 m, and is identical for every model including those that consume no patch. The training pool is likewise capped per class at a single protocol-level constant set by the most restrictive model, so that the full-budget point is the same experiment for every model rather than a different one for each."

**Reason.** Two consequences that are easy to miss and each of which would invalidate results. First, the patch is a leakage channel that block partitioning alone does not close, since the patch footprint is a physical neighbourhood around the test parcel and blocks are only a partition of locations; the buffer closes it, symmetrically, but only if its width is derived from the patch geometry rather than chosen as a round number. Second, if the pool cap were set per model, the patch models would be trained at the full budget on far fewer parcels than the embedding models, because patch extraction is roughly two orders of magnitude more expensive per parcel, so the full-budget comparison would confound the encoder with the training-set size. Setting one cap for all models costs the embedding models some ceiling and buys a valid comparison.

---

## Delta 23. The shipped time series is not the only available input

**Proposal, line 131, and `CLAUDE.md`, Datasets section.**

Both documents describe EuroCropsML's per-parcel median time series as though it were the input to Phase 1. It is one of six declared input representations. Raw Sentinel-2 imagery is obtainable for the parcel footprints, so patch-based models are fully in scope, and the input format places no constraint on model selection.

**Addition to line 131, after the first sentence:**
> "That time series is one of the input representations used in Phase 1 and not the only one. Raw Sentinel-2 imagery is downloaded for the parcel footprints where a model requires it, so the input format does not constrain the choice of model."

**Reason.** Left as written, both documents imply that patch-based models must be shoehorned into a one-by-one pixel time series, which would either exclude TerraMind and THOR from the study or evaluate them on a degenerate input. It also obscures the representation factor of Delta 21, since if there were only one input there would be no factor to control.

---

## Delta 24. CLAUDE.md needs the corresponding edits

**Severity: housekeeping.**

1. **Datasets section.** "Built-in K-shot protocol and cross-country transfer protocol." Replace with the corrected description from Delta 2: the built-in protocol targets Estonia only, from Latvia and optionally Portugal, and applies no spatial control.
2. **Phase 1 section, the "Open inconsistency to resolve before WP2" paragraph.** Replace the statement of the problem with the statement of the resolution: K is samples per class, on the grid `{1, 5, 10, 20, 50, 100, 200, 500, pool}`, applied everywhere, per `docs/phase1/protocol.md`.
3. **Phase 1 section, the Protocol paragraph.** "the 15 to 20 most frequent crops per country" should become the eligibility rule of Delta 4, and "spatial block cross-validation (Roberts et al., 2017)" should become the concrete scheme of Delta 3. Add a pointer: "The authoritative Phase 1 specification is `docs/phase1/protocol.md`."
4. **Selection of GFMs section.** The table should gain an **input representation** column, since the four models consume different inputs and that is a Phase 1 experimental factor and not only a Phase 2 constraint. The sentence "The weights-versus-embeddings split is not merely logistical: it sets the ceiling on explainability" should be extended to note that it also fixes the input representation and the spatial support, and therefore bears on the validity of the Phase 1 comparison, per Delta 21.
5. **Datasets section.** The description of EuroCropsML as providing a per-parcel median time series should note that raw Sentinel-2 imagery is downloaded for the parcel footprints where a model requires it, so the shipped time series is one input representation among several and the input format does not constrain model selection, per Delta 23.

**Reason.** `CLAUDE.md` states that the proposal wins where the two disagree. Phase 1 now has a third document that is more specific than either, so the precedence order should be stated: `docs/phase1/protocol.md` for Phase 1, the proposal elsewhere.
