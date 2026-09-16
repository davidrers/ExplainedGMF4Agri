"""Deterministic, context-free random number generation.

Every stochastic decision in the Phase 1 protocol is derived from a string of
context parts rather than from a mutable global generator.  The consequences
that the protocol depends on are:

* the support set drawn at a given ``(country, class, K, draw_seed)`` is the
  same regardless of which foundation model, which head, or which machine
  requests it, so the comparison between models is never confounded by a
  difference in the labels they were given;
* results do not depend on the order in which classes, countries or budgets
  happen to be iterated, so a partially completed sweep can be resumed or
  parallelised without changing any answer.
"""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["derived_seed", "derived_rng"]

_SEPARATOR = "\x1f"


def derived_seed(*parts: object) -> int:
    """Derive a 128-bit integer seed from a tuple of context parts.

    Args:
        *parts: Context identifying the draw, for example ``("support", 0,
            "EE", "3301010101", 7)``.  Parts are converted with :func:`str`
            and joined with a separator that cannot occur in the identifiers
            used by this project, so distinct tuples cannot collide by
            concatenation.

    Returns:
        A non-negative integer suitable as ``numpy`` seed entropy.
    """
    payload = _SEPARATOR.join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=16).digest(), "big")


def derived_rng(*parts: object) -> np.random.Generator:
    """Return a :class:`numpy.random.Generator` determined by the context parts.

    Args:
        *parts: Context identifying the draw, as for :func:`derived_seed`.

    Returns:
        A fresh PCG64 generator.  Two calls with equal parts return generators
        that produce identical streams.
    """
    return np.random.default_rng(np.random.SeedSequence(derived_seed(*parts)))
