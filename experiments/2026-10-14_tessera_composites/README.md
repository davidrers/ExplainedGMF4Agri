# TESSERA driven from our monthly composites, frozen and fine-tuned, 14 October 2026

Nothing in the main workflow (`src/`, `configs/`, `scripts/`, `tests/`) is edited. The study runs in two stages.
- **Stage 1**, a feasibility spike, is done. It lives in `spike/` and is reported in [REPORT.md](REPORT.md).
- **Stage 2**, the fit with the encoder frozen or trained, is designed in [DESIGN.md](DESIGN.md) and built in this
  folder. Its pilot on `EE_2021_rev_mini` is running.

**Question.** Can the TESSERA encoder be run on the twelve monthly Sentinel-2 and Sentinel-1 composites that every
other arm already reads, instead of on its native per-acquisition series, so that it can be fine-tuned without
downloading new imagery? And once it is: does fine-tuning it help at each label budget, against the same encoder
frozen?

Today's TESSERA arm reads precomputed embeddings, which come from the v1.0 store on Source Cooperative. That
product cannot be fine-tuned. To fine-tune it, the encoder has to run inside our fit, and the only input on disk
for that is the composites. TESSERA's own preprocessing needs every acquisition of the year, about 1.2 TB for
Estonia, and the user ruled that download out on 14 October 2026.

**Stage 1, the spike.** Does the encoder still separate crops when fed composites?
- **What ran:** the TESSERA v1.1 encoder (AWS and MPC checkpoints) and the four v2 students (Nano, Small, Medium,
  Large) were run, frozen, on the composites of `EE_2021_rev_mini` (57 chips, 5 blocks).
- **What they were compared with:** the native-series embeddings of the v1.0 store (our current arm) and of the
  v1.1 store. The v1.1 store was built by the same AWS v1.1 checkpoint, so the AWS comparison varies the input
  alone.
- **Measures:** per-pixel cosine to the native embedding, a ridge map onto it, and a per-pixel MLP probe (the
  arm's head) in leave-one-block-out cross-validation, 3 seeds.

**Stage 2, the fit.** [DESIGN.md](DESIGN.md) holds the decisions and their reasons. In short:
- **Arms:** TESSERA v2 Medium on the composites, frozen (`tessera_v2m_frozen`) or fine-tuned (`tessera_v2m_full`).
  The two differ in `model.trainable` alone, and live in `arms_run/`.
- **Head and sampling:** a per-pixel MLP head; up to 64 pixels per support parcel per epoch, the same in both arms.
- **Protocol:** K = 100, 20 and 5 %, draw 0, seeds 0 to 2, 15 epochs, checkpoint at the lowest validation loss.

The code is in this folder:
- `tessera_v2.py`: the encoder, copied from TESSERA under MIT, with a gradient-enabled batch encoder;
- `pixels.py`: the pixel store and the frozen embedding cache;
- `fit_pixel.py`: one cell;
- `run.py`: the entry, with `scripts/run_kshot.py`'s command line;
- `check.py`: the design's checks, all passed on 14 October 2026.

**How to run stage 2.**

    poetry run python experiments/2026-10-14_tessera_composites/check.py
    poetry run python experiments/2026-10-14_tessera_composites/run.py -e experiments/2026-10-14_tessera_composites/experiment.yaml --chips data/eurocrops_chips/EE_2021_rev_mini
    bash scripts/cluster/submit.sh experiments/2026-10-14_tessera_composites/experiment.yaml EE_2021_rev    # cluster, one job per arm
    poetry run python experiments/2026-10-14_tessera_composites/report.py --chips EE_2021_rev_mini

**How to run stage 1.** From the repository root, after recreating the git-ignored inputs below:

    poetry run python experiments/2026-10-14_tessera_composites/spike/fetch_store.py        # v1.1 store onto the chips, ~1 h
    poetry run python experiments/2026-10-14_tessera_composites/spike/embed_composites.py   # v1.1 and v2 on the composites
    poetry run python experiments/2026-10-14_tessera_composites/spike/compare.py            # metrics.json
    poetry run python experiments/2026-10-14_tessera_composites/spike/figures.py            # figures/spike_*.png

**Git-ignored inputs.**
- **TESSERA's code** is a clone pinned at commit `e368539`:
  `git clone https://github.com/ucam-eo/tessera data/external/tessera && git -C data/external/tessera checkout e368539f7c4711e5f16fdd09801573830e34baf9`.
- **Weights.** They are in `data/weights/tessera/`, with `SHA256SUMS` beside them:
  - v1.1 encoder-only checkpoints, from the Google Drive links of TESSERA's README, in `v1_1/`;
  - the v2 students, from Hugging Face `geotessera/TESSERA-V-2.0-2B-{N,S,M,L}` (`ckpt/student_<size>.pt`), in `v2/`.
  - All six pickles hold tensors and plain containers only, so they load with `torch.load(weights_only=True)`.

**Results.** `results/2026-10-14_tessera_composites/spike/`:
- `emb/<representation>/<chip>.npy`: float16 (128, 224, 224);
- `metrics.json`;
- the logs.

**Outcome.** Stage 1: yes ([REPORT.md](REPORT.md)).
- **Probe:** frozen TESSERA on our composites separates crops on the pilot about as well as from its native series.
  Probe Macro-F1 is 0.371 ± 0.003 for v2 Medium on composites, against 0.370 (v1.1 native) and 0.373 (v1.0 native,
  the current arm). Every composite variant lands within 0.018.
- **Embedding:** the vector itself moves. The median cosine to the native embedding is 0.76 for the same v1.1
  checkpoint.
- **Per class:** composites lose on peas, potatoes and harvested green legumes, and gain on maize.
- **Consequence:** stage 2 can fine-tune on data already on disk, with v2 Medium as the default encoder.
