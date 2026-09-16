"""Data-driven estimation of the spatial autocorrelation length of crop labels.

The block size used for spatial blocking must be chosen from the data rather
than picked by hand.  Crop labels are categorical, so an ordinary semivariogram
does not apply directly.  The estimator implemented here is the categorical
analogue: the *label agreement decay*, that is the excess probability that two
parcels separated by a distance :math:`d` carry the same crop label, above the
agreement expected by chance.

Writing :math:`p_0 = \\sum_c p_c^2` for the chance agreement under the marginal
class distribution, the estimator forms

.. math::

    E(d) = \\Pr[\\,y_i = y_j \\mid \\lVert s_i - s_j \\rVert \\approx d\\,] - p_0,

and fits :math:`E(d) = E_0 \\exp(-d / a)`.  The quantity reported is the
*practical range* :math:`R = 3a`, at which the excess agreement has decayed to
five per cent of its value at the origin.  This is the direct categorical
counterpart of the practical range of an exponential variogram and is exactly
the quantity that governs leakage between neighbouring parcels, so it is the
natural block size.

The estimator is pure NumPy and fully determined by its seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["AgreementDecay", "estimate_label_range_m", "label_agreement_decay"]


@dataclass(frozen=True)
class AgreementDecay:
    """Result of a label-agreement decay estimation.

    Attributes:
        chance_agreement: :math:`p_0`, the agreement expected under
            independence given the observed marginal class frequencies.
        bin_centres_m: Centre of each distance bin, in metres.
        excess_agreement: :math:`E(d)` in each bin.
        n_pairs: Number of parcel pairs contributing to each bin.
        decay_length_m: The fitted :math:`a`, or ``None`` if the fit failed.
        practical_range_m: :math:`3a`, or ``None`` if the fit failed.
        r_squared: Coefficient of determination of the log-linear fit.
        n_bins_used: Number of bins that entered the fit.
        note: Human-readable status, stamped into the split manifest.
    """

    chance_agreement: float
    bin_centres_m: np.ndarray
    excess_agreement: np.ndarray
    n_pairs: np.ndarray
    decay_length_m: float | None
    practical_range_m: float | None
    r_squared: float | None
    n_bins_used: int
    note: str

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation."""
        return {
            "chance_agreement": float(self.chance_agreement),
            "bin_centres_m": [float(v) for v in self.bin_centres_m],
            "excess_agreement": [float(v) for v in self.excess_agreement],
            "n_pairs": [int(v) for v in self.n_pairs],
            "decay_length_m": None if self.decay_length_m is None else float(self.decay_length_m),
            "practical_range_m": (
                None if self.practical_range_m is None else float(self.practical_range_m)
            ),
            "r_squared": None if self.r_squared is None else float(self.r_squared),
            "n_bins_used": int(self.n_bins_used),
            "note": self.note,
        }


def label_agreement_decay(
    x_m: np.ndarray,
    y_m: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = 0,
    pool_size: int = 40_000,
    n_anchors: int = 3_000,
    d_min_m: float = 250.0,
    d_max_m: float = 60_000.0,
    n_bins: int = 24,
    min_pairs_per_bin: int = 200,
    min_excess_at_short_range: float = 0.02,
    n_sigma: float = 3.0,
    min_r_squared: float = 0.5,
    chunk: int = 256,
) -> AgreementDecay:
    """Estimate the label-agreement decay curve and its exponential range.

    Restricting the fit to bins of positive excess would, on a spatially
    random label field, retain only the upward half of the sampling noise and
    manufacture an apparent decay.  Two guards prevent that.  The excess in
    the shortest well-populated bin must exceed both an absolute floor and
    ``n_sigma`` binomial standard errors, and the log-linear fit must reach
    ``min_r_squared``.  When either guard fails, no range is reported and the
    caller falls back to a configured block size.

    Args:
        x_m: Projected eastings of the parcels, in metres.
        y_m: Projected northings of the parcels, in metres.
        labels: Integer or string class labels, one per parcel.
        seed: Seed controlling the subsampling of the anchor and pool sets.
        pool_size: Maximum number of parcels retained as the comparison pool.
            Subsampling is uniform, so the pair-distance distribution is
            preserved up to Monte Carlo error.
        n_anchors: Number of anchor parcels drawn from the pool.
        d_min_m: Lower edge of the first distance bin, in metres.
        d_max_m: Upper edge of the last distance bin, in metres.
        n_bins: Number of logarithmically spaced distance bins.
        min_pairs_per_bin: Bins with fewer pairs are excluded from the fit.
        min_excess_at_short_range: Absolute floor on the excess agreement in
            the shortest well-populated bin.
        n_sigma: Number of binomial standard errors the short-range excess
            must exceed.
        min_r_squared: Floor on the coefficient of determination of the fit.
        chunk: Number of anchors processed per vectorised block.

    Returns:
        An :class:`AgreementDecay`.  ``practical_range_m`` is ``None`` when the
        curve is too noisy or too flat to support a fit, in which case the
        caller must fall back to a configured default block size.
    """
    x_m = np.asarray(x_m, dtype=np.float64).ravel()
    y_m = np.asarray(y_m, dtype=np.float64).ravel()
    codes = np.unique(np.asarray(labels).ravel(), return_inverse=True)[1].astype(np.int64)
    n = x_m.size
    if n != y_m.size or n != codes.size:
        raise ValueError("x_m, y_m and labels must have the same length")

    rng = np.random.default_rng(np.random.SeedSequence(int(seed)))
    if n > pool_size:
        sel = np.sort(rng.choice(n, size=pool_size, replace=False))
        x_m, y_m, codes = x_m[sel], y_m[sel], codes[sel]
        n = pool_size

    counts = np.bincount(codes)
    probs = counts / counts.sum()
    p0 = float(np.sum(probs * probs))

    if n < 50:
        empty = np.zeros(0)
        return AgreementDecay(p0, empty, empty, empty.astype(int), None, None, None, 0,
                              "too few parcels to estimate a range")

    n_anchors = int(min(n_anchors, n))
    anchor_idx = np.sort(rng.choice(n, size=n_anchors, replace=False))

    edges = np.geomspace(float(d_min_m), float(d_max_m), n_bins + 1)
    same_sum = np.zeros(n_bins, dtype=np.float64)
    pair_sum = np.zeros(n_bins, dtype=np.float64)

    for start in range(0, n_anchors, chunk):
        idx = anchor_idx[start : start + chunk]
        dx = x_m[idx][:, None] - x_m[None, :]
        dy = y_m[idx][:, None] - y_m[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        same = (codes[idx][:, None] == codes[None, :]).astype(np.float64)
        # Exclude each anchor's self-pair.
        same[np.arange(idx.size), idx] = 0.0
        dist[np.arange(idx.size), idx] = np.inf

        bin_idx = np.digitize(dist.ravel(), edges) - 1
        flat_same = same.ravel()
        valid = (bin_idx >= 0) & (bin_idx < n_bins)
        pair_sum += np.bincount(bin_idx[valid], minlength=n_bins).astype(np.float64)
        same_sum += np.bincount(bin_idx[valid], weights=flat_same[valid], minlength=n_bins)

    centres = np.sqrt(edges[:-1] * edges[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        agreement = np.where(pair_sum > 0, same_sum / pair_sum, np.nan)
    excess = agreement - p0

    populated = (pair_sum >= min_pairs_per_bin) & np.isfinite(excess)
    if not populated.any():
        return AgreementDecay(p0, centres, excess, pair_sum.astype(int), None, None, None, 0,
                              "no sufficiently populated distance bin")

    # Guard one: the shortest well-populated bin must show a real excess,
    # otherwise the label field is spatially random and any apparent decay is
    # sampling noise.
    first = int(np.flatnonzero(populated)[0])
    se = float(np.sqrt(max(p0 * (1.0 - p0), 1e-12) / pair_sum[first]))
    threshold = max(float(min_excess_at_short_range), float(n_sigma) * se)
    if not (excess[first] > threshold):
        return AgreementDecay(
            p0, centres, excess, pair_sum.astype(int), None, None, None, 0,
            f"short-range excess {excess[first]:.4f} does not exceed {threshold:.4f}",
        )

    usable = populated & (excess > 1e-4)
    if int(usable.sum()) < 4:
        return AgreementDecay(p0, centres, excess, pair_sum.astype(int), None, None, None,
                              int(usable.sum()), "fewer than four usable bins")

    d = centres[usable]
    log_e = np.log(excess[usable])
    w = pair_sum[usable]
    # Weighted least squares of log(E) on d.
    sw = w.sum()
    d_bar = float((w * d).sum() / sw)
    e_bar = float((w * log_e).sum() / sw)
    s_dd = float((w * (d - d_bar) ** 2).sum())
    s_de = float((w * (d - d_bar) * (log_e - e_bar)).sum())
    if s_dd <= 0.0:
        return AgreementDecay(p0, centres, excess, pair_sum.astype(int), None, None, None,
                              int(usable.sum()), "degenerate distance support")
    slope = s_de / s_dd
    if slope >= 0.0:
        return AgreementDecay(p0, centres, excess, pair_sum.astype(int), None, None, None,
                              int(usable.sum()), "no decay detected (non-negative slope)")

    pred = e_bar + slope * (d - d_bar)
    ss_res = float((w * (log_e - pred) ** 2).sum())
    ss_tot = float((w * (log_e - e_bar) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Guard two: a weak fit is not evidence of an exponential decay.
    if not np.isfinite(r2) or r2 < float(min_r_squared):
        return AgreementDecay(
            p0, centres, excess, pair_sum.astype(int), None, None,
            None if not np.isfinite(r2) else float(r2), int(usable.sum()),
            f"log-linear fit too weak (R2 = {r2:.3f} < {min_r_squared:.2f})",
        )

    a = -1.0 / slope
    return AgreementDecay(
        chance_agreement=p0,
        bin_centres_m=centres,
        excess_agreement=excess,
        n_pairs=pair_sum.astype(int),
        decay_length_m=float(a),
        practical_range_m=float(3.0 * a),
        r_squared=r2,
        n_bins_used=int(usable.sum()),
        note="ok",
    )


def estimate_label_range_m(
    x_m: np.ndarray,
    y_m: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = 0,
    min_block_m: float = 2_000.0,
    max_block_m: float = 50_000.0,
    round_to_m: float = 1_000.0,
    fallback_m: float = 10_000.0,
    **kwargs: object,
) -> tuple[float, AgreementDecay]:
    """Estimate a block edge length from the label-agreement decay.

    The practical range is rounded up to a multiple of ``round_to_m`` and
    clipped to ``[min_block_m, max_block_m]``, so that the chosen block size is
    a round number that a reader can sanity-check, and so that a pathological
    fit cannot produce an absurd tessellation.

    Args:
        x_m: Projected eastings, in metres.
        y_m: Projected northings, in metres.
        labels: Class labels, one per parcel.
        seed: Seed for the subsampling.
        min_block_m: Lower clip on the returned block size.
        max_block_m: Upper clip on the returned block size.
        round_to_m: Rounding granularity, in metres.
        fallback_m: Block size used when no range can be estimated.
        **kwargs: Forwarded to :func:`label_agreement_decay`.

    Returns:
        The chosen block edge length in metres and the full diagnostic object.
    """
    decay = label_agreement_decay(x_m, y_m, labels, seed=seed, **kwargs)  # type: ignore[arg-type]
    raw = decay.practical_range_m if decay.practical_range_m is not None else float(fallback_m)
    rounded = float(np.ceil(raw / round_to_m) * round_to_m)
    return float(np.clip(rounded, min_block_m, max_block_m)), decay
