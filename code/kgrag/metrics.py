"""Answer, retrieval and groundedness metrics, plus paired bootstrap significance."""
import collections

import numpy as np

from .text import normalize_answer, content_tokens


def exact_match(pred, gold):
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1(pred, gold):
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    ns = sum(common.values())
    if not p or not g or ns == 0:
        return float(p == g)
    pr, rc = ns / len(p), ns / len(g)
    return 2 * pr * rc / (pr + rc)


def recall_at_k(retrieved, gold_pids):
    """Fraction of supporting passages among the retrieved ones."""
    return len(set(retrieved) & set(gold_pids)) / max(1, len(gold_pids))


def all_support_at_k(retrieved, gold_pids):
    """1 if every supporting passage is retrieved (needed for a correct multi-hop answer)."""
    return float(set(gold_pids) <= set(retrieved))


def support_rate(pred, context):
    """Groundedness proxy: share of the answer's content tokens that occur in the retrieved context.

    This is a lexical proxy, not an NLI-based faithfulness score.
    """
    a = content_tokens(pred)
    if not a:
        return 0.0
    ctx = set(content_tokens(context))
    return sum(t in ctx for t in a) / len(a)


def unsupported(pred, context, thr=0.8):
    """1 if the answer is not (lexically) supported by the retrieved context: a hallucination proxy."""
    return float(support_rate(pred, context) < thr)


def paired_bootstrap(a, b, n=10000, seed=0):
    """Two-sided paired bootstrap test on the mean difference mean(a) - mean(b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = a - b
    obs = d.mean()
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    centred = means - means.mean()
    p = (np.abs(centred) >= abs(obs)).mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(obs), float(lo), float(hi), float(max(p, 1.0 / n))
