"""Numerical verification of the two propositions in Section 4 of the paper."""
import itertools
import os
import sys
import unittest

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kgrag.graph import EntityGraph  # noqa: E402


def random_graph(seed, n=12, m=15, density=0.25):
    rng = np.random.default_rng(seed)
    A = (rng.random((n, m)) < density) * rng.uniform(0.5, 2.0, (n, m))
    for i in range(n):  # no dangling passages / entities (matches the construction in the paper)
        if A[i].sum() == 0:
            A[i, rng.integers(m)] = 1.0
    for j in range(m):
        if A[:, j].sum() == 0:
            A[rng.integers(n), j] = 1.0
    W = np.block([[np.zeros((n, n)), A], [A.T, np.zeros((m, m))]])
    P = W / W.sum(axis=1, keepdims=True)
    return n, m, P


def exact_ppr(P, s, alpha):
    N = P.shape[0]
    return alpha * s @ np.linalg.inv(np.eye(N) - (1 - alpha) * P)


class TestPPR(unittest.TestCase):
    def test_path_lower_bound(self):
        """pi(v_t) >= alpha (1-alpha)^t s(v_0) prod P(v_{i-1}, v_i) for every walk."""
        for seed in range(5):
            n, m, P = random_graph(seed)
            alpha = 0.15
            for q in range(n, n + 3):  # entity seeds
                s = np.zeros(n + m)
                s[q] = 1.0
                pi = exact_ppr(P, s, alpha)
                for p1 in range(n):
                    if P[q, p1] == 0:
                        continue
                    for e in range(n, n + m):
                        if P[p1, e] == 0:
                            continue
                        for p2 in range(n):
                            if P[e, p2] == 0:
                                continue
                            bound = alpha * (1 - alpha) ** 3 * P[q, p1] * P[p1, e] * P[e, p2]
                            self.assertGreaterEqual(pi[p2] + 1e-12, bound)

    def test_power_iteration_error_bound(self):
        """||pi_T - pi||_1 <= 2 (1-alpha)^T for the iteration pi_0 = s."""
        for alpha in (0.15, 0.3, 0.5):
            n, m, P = random_graph(7)
            s = np.zeros(n + m)
            s[n + 2] = 1.0
            pi = exact_ppr(P, s, alpha)
            cur = s.copy()
            for T in range(1, 25):
                cur = alpha * s + (1 - alpha) * cur @ P
                self.assertLessEqual(np.abs(cur - pi).sum(), 2 * (1 - alpha) ** T + 1e-12)

    def test_graph_class_rows_stochastic(self):
        passages = [
            {"title": "Alpha Film", "text": "Alpha Film was directed by Bob Smith."},
            {"title": "Bob Smith", "text": "Bob Smith was born in Paris."},
            {"title": "Paris", "text": "Paris is a city in France."},
        ]
        g = EntityGraph(passages, idf=False, max_df_frac=None)
        rows = np.asarray(g.P.sum(axis=1)).ravel()
        self.assertTrue(np.allclose(rows, 1.0))
        pi = g.ppr(np.eye(g.n + g.m)[g.n + g.vocab["alpha film"]])
        self.assertGreater(pi[1], pi[2])  # one hop nearer than two hops


if __name__ == "__main__":
    unittest.main()
