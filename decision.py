"""
Turning model scores into costed decisions. Shared by train.py and app.py.

A case is one hive photo. Its score is the highest varroa confidence the model
finds in it. The truth is whether the photo has at least one labelled mite.

Two-way policy:  score >= t -> treat, else leave alone.
Three-way policy: score < low -> pass (leave alone)
                  score >= high -> fail (treat)
                  otherwise -> a person checks, and is assumed to get it right.

Costs are expected euros per 1,000 cases at an ASSUMED base rate, not at the
test set's own share of positives. Rates (TPR, FPR, review shares) come from
the test set; the mix of positives and negatives comes from the business.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Costs:
    fn: float
    fp: float
    tp: float
    tn: float
    review: float
    base_rate: float

    @classmethod
    def from_config(cls, cfg) -> "Costs":
        return cls(cfg.COST_FN, cfg.COST_FP, cfg.COST_TP, cfg.COST_TN,
                   cfg.COST_REVIEW, cfg.BASE_RATE)


def thresholds(step: float = 0.01) -> np.ndarray:
    return np.round(np.arange(0.0, 1.0 + 1e-9, step), 4)


def three_way(scores, y, low, high) -> dict:
    """Shares of positives / negatives falling in each outcome bucket."""
    scores, y = np.asarray(scores, float), np.asarray(y, bool)
    low, high = min(low, high), max(low, high)
    passed = scores < low
    failed = scores >= high
    review = ~passed & ~failed
    pos, neg = max(y.sum(), 1), max((~y).sum(), 1)
    return {
        # counts on the test set (rows: actual, cols: decision)
        "pos_pass": int((passed & y).sum()), "pos_review": int((review & y).sum()),
        "pos_fail": int((failed & y).sum()),
        "neg_pass": int((passed & ~y).sum()), "neg_review": int((review & ~y).sum()),
        "neg_fail": int((failed & ~y).sum()),
        # rates within each actual class
        "p_pass": (passed & y).sum() / pos, "p_review": (review & y).sum() / pos,
        "p_fail": (failed & y).sum() / pos,
        "n_pass": (passed & ~y).sum() / neg, "n_review": (review & ~y).sum() / neg,
        "n_fail": (failed & ~y).sum() / neg,
    }


def cost_per_1000(r: dict, c: Costs) -> dict:
    """Expected cost per 1,000 cases and the review share, at the assumed base rate."""
    pi = c.base_rate
    # Positives: passed = missed (FN); reviewed = caught by person then treated; failed = TP
    pos = r["p_pass"] * c.fn + r["p_review"] * (c.review + c.tp) + r["p_fail"] * c.tp
    # Negatives: passed = TN; reviewed = person clears it; failed = FP (treated for nothing)
    neg = r["n_pass"] * c.tn + r["n_review"] * (c.review + c.tn) + r["n_fail"] * c.fp
    review_share = pi * r["p_review"] + (1 - pi) * r["n_review"]
    return {
        "cost_per_1000": 1000 * (pi * pos + (1 - pi) * neg),
        "review_share": review_share,
        "missed_per_1000": 1000 * pi * r["p_pass"],
        "false_treat_per_1000": 1000 * (1 - pi) * r["n_fail"],
    }


def sweep_single(scores, y, c: Costs, step=0.01):
    """Two-way policy: cost against threshold. Returns list of dict rows."""
    rows = []
    for t in thresholds(step):
        r = three_way(scores, y, t, t)
        out = cost_per_1000(r, c)
        rows.append({"threshold": float(t),
                     "recall": r["p_fail"], "fpr": r["n_fail"],
                     "TP": r["pos_fail"], "FN": r["pos_pass"],
                     "FP": r["neg_fail"], "TN": r["neg_pass"], **out})
    return rows


def best_band(scores, y, c: Costs, step=0.01, max_review_share: float | None = None):
    """Grid search over (low, high) with low <= high for the cheapest three-way policy."""
    ts = thresholds(step)
    best = None
    for i, lo in enumerate(ts):
        for hi in ts[i:]:
            out = cost_per_1000(three_way(scores, y, lo, hi), c)
            if max_review_share is not None and out["review_share"] > max_review_share + 1e-12:
                continue
            key = (round(out["cost_per_1000"], 6), out["review_share"])
            if best is None or key < best[0]:
                best = (key, float(lo), float(hi), out)
    _, lo, hi, out = best
    return lo, hi, out


def alternatives(c: Costs) -> dict:
    """Cost per 1,000 of policies that do not need a model."""
    pi = c.base_rate
    return {
        "Do nothing (never treat)": 1000 * (pi * c.fn + (1 - pi) * c.tn),
        "Treat every hive": 1000 * (pi * c.tp + (1 - pi) * c.fp),
        "Person checks every hive": 1000 * (c.review + pi * c.tp + (1 - pi) * c.tn),
    }


def staffing(review_share: float, cfg) -> dict:
    cases_per_day = cfg.N_HIVES * cfg.CHECKS_PER_HIVE_PER_WEEK / cfg.WORKDAYS_PER_WEEK
    reviews = cases_per_day * review_share
    hours = reviews * cfg.REVIEW_MINUTES / 60
    return {"cases_per_day": cases_per_day, "reviews_per_day": reviews,
            "review_hours_per_day": hours,
            "people_needed": hours / cfg.PRODUCTIVE_HOURS_PER_DAY}
