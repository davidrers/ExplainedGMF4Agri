# Spatial and temporal transferability: planning material

Gathered on 26 and 27 October 2026 so that the transfer experiments can be planned in a separate session. Nothing
in this folder is a decision: decisions go into `docs/phase1/protocol.md` once the user agrees, and an agreed
design into `docs/superpowers/specs/<date>-transferability-design.md`, with its implementation plan beside it in
`docs/superpowers/plans/`, as for the earlier designs.

## In this folder

| File | Holds |
|---|---|
| [eurocrops_coverage.md](eurocrops_coverage.md) | Which countries, regions and years EuroCrops v11 and v2.0 hold, what the repository already has, the v2 file layout, and what that means for spatial and temporal transfer |
| [eurocrops_v2_files.csv](eurocrops_v2_files.csv) | The 153 yearly and 17 stack files of EuroCrops v2.0 (`gpqtv202/`, listed 26 October 2026) |
| [gaez_v5_and_zonings.md](gaez_v5_and_zonings.md) | GAEZ v5: access, inputs and resolution, the 57-zone class structure (thermal regime, moisture, soil, special classes), how it bears on the design; and Köppen-Geiger, the Environmental Stratification of Europe, the EEA regions and the UNESCO aridity map compared |

## What is already written down elsewhere

- **`docs/phase1/protocol.md` section 5** and decisions **D13 to D16**: six ordered pairs among EE, LV and PT and
  three many-to-one settings; three regimes (zero-shot, target only, source plus K target labels); the head
  retrained on the intersected label space; Portugal on a coarsened HCAT level as a low-overlap regime; class prior
  shift separated from representation failure by per-class F1, a prior-corrected readout, a prior-matched test
  subsample, and the transfer gap plotted against prior divergence and phenological distance. Section 5 and the fit
  budget of section 6.4 were written for the per-parcel heads and feature-set groups of protocol version 1.0, so
  they need restating for frozen encoders with segmentation decoders.
- **`docs/phase1/proposal_deltas.md`**, deltas 13 to 15: why the transfer design departs from the proposal.
- **Decided on 2 October 2026, not yet in the protocol:** cross-country transfer only within one GAEZ v5 zone
  (`figures/phase1_experimental_setup_prompt.md`).
- **`docs/research/prior_gfm_use_eurocropsml.md`**: the published Latvia to Estonia and Latvia plus Portugal to
  Estonia numbers, where adding Portugal hurt Estonia.
- **Status of the countries (27 October 2026):** Estonia 2021 runs end to end. `LV_2021` has its Sentinel-2 and
  Sentinel-1 chips (11,500, exported 18 to 23 October 2026); its split and embeddings are not built. Lithuania and
  Portugal 2021 exist as v11 vectors only. The third country is the user's open choice.

## Ideas raised on 26 October 2026, not decided

From the discussion in the session that gathered this material. Starting points for the planning, nothing more.

1. **Two label levels.** Each country keeps its fine classes for the in-country curves of RQ1; every fine class
   also maps to one group of a fixed EU list (about 12 to 15 HCAT groups plus "other crops"), fixed before results
   are seen, on which transfer runs. Fine classes selected by test support (for example at least 100 test parcels
   over at least 10 test blocks) rather than by total count.
2. **Budgets in transfer as parcels per class.** K % stays for the in-country curves, but 5 % of Lithuania is about
   ten times 5 % of Estonia, so target budgets in transfer are absolute counts per class, capped by the pool, with a
   readout corrected to the target's crop shares. The current draw is proportional (`draw_support`,
   `src/gfm4agri/benchmark/segmentation_data.py`); `configs/class_scheme_eurocropsml.yaml` still says K is an
   absolute count, which contradicts the code.
3. **A ladder of transfer distances:** spatial blocks within a country (have); a whole region held out within a
   country; across countries within one zone; across zones. The drop from rung to rung says whether region,
   national labelling or climate breaks transfer.
4. **Zone × country as a 2 × 2.** Same zone and different country (EE, LV, LT) changes only labelling conventions
   and farm structure; same country and different zone (a country spanning two zones, such as Austria or France)
   changes only climate. The two off-diagonal cells isolate one factor each. LV to EE sits in the first, if the
   layers confirm that both countries fall in one zone; PT to the Baltics changes both factors at once.
5. **Zones from GAEZ components:** every 4 x 4 block tagged with its temperature regime and moisture class, a second
   zoning tagged alongside, transfer distance measured on continuous climate variables, a buffer at zone borders.
6. **Time as a second axis (new on 27 October).** EuroCrops v2 holds 3 to 16 consecutive years per region (Estonia
   2015 to 2023), and its stack files follow each field across years. Training on one year and testing on another
   in the same place changes only the weather and the rotation, which is the cleanest climate shift available, and
   it can be placed on the same continuous climate axis using AgERA5 for each year. Open: whether v2's Estonia 2021
   matches v11's, which years every arm's imagery and products cover, and how HCAT4 maps to our classes.

## Questions the planning session should settle

- Spatial, temporal or both, and which is the thesis's experiment and which is further work.
- The third country, and whether v11 (Latvia, Lithuania, 2021 only) or v2 (multi-year, no Latvia or Lithuania) is
  the label source for transfer.
- The label space of transfer: pairwise intersection (protocol D14) or a fixed EU list.
- The zoning and its unit (block, chip or parcel), and whether the 2 October rule "within one zone only" stays a
  rule or becomes an axis.
- The fit budget: which arms and budgets enter transfer, given that each transfer setting refits the decoders.
