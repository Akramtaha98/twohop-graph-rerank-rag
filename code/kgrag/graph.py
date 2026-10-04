"""Passage-entity bipartite graph and Personalized PageRank (PPR).

Nodes: n passages followed by m entities. Edge (p, e) exists when entity e occurs in passage p
(the passage title is always added as an entity, so no passage and no entity is isolated and
the transition matrix has no dangling rows).
"""
import math
from collections import defaultdict

import numpy as np
import scipy.sparse as sp

from .text import ekey, extract_entities, tokens, STOP


class EntityGraph:
    def __init__(self, passages, ner="regex", idf=True, max_df_frac=0.2):
        """passages: list of dict(title, text)."""
        self.n = len(passages)
        self.ner = ner
        vocab, rows, cols, df = {}, [], [], defaultdict(int)
        for pid, p in enumerate(passages):
            ents = {ekey(e) for e in extract_entities(p["text"], ner)}
            ents.add(ekey(p["title"]))
            ents.discard("")
            for e in ents:
                if e not in vocab:
                    vocab[e] = len(vocab)
                df[e] += 1
                rows.append(pid)
                cols.append(vocab[e])
        self.vocab = vocab
        self.m = len(vocab)
        self.df = np.zeros(self.m)
        for e, j in vocab.items():
            self.df[j] = df[e]
        w = np.ones(len(rows))
        if idf:
            w = np.array([math.log(1.0 + self.n / self.df[c]) for c in cols])
        # Hub suppression: entities occurring in more than max_df_frac of passages carry no linking signal.
        if max_df_frac is not None:
            keep = np.array([self.df[c] <= max(2, max_df_frac * self.n) for c in cols])
            for k in np.where(~keep)[0]:
                w[k] = 0.0
        A = sp.csr_matrix((w, (rows, cols)), shape=(self.n, self.m))
        A.eliminate_zeros()
        self.A = A
        self.Ac = A.tocsc()
        # Row-stochastic transition matrix over the n+m node bipartite graph.
        W = sp.bmat([[None, A], [A.T, None]], format="csr")
        deg = np.asarray(W.sum(axis=1)).ravel()
        deg[deg == 0] = 1.0  # isolated (hub-removed) nodes keep zero rows; PPR restarts handle the leak
        self.P = sp.diags(1.0 / deg) @ W
        self.PT = self.P.T.tocsr()
        self.max_ngram = 4

    def query_entities(self, question):
        """Entities from the vocabulary that appear as token n-grams of the question."""
        toks = tokens(question)
        found = []
        for L in range(self.max_ngram, 0, -1):
            for i in range(len(toks) - L + 1):
                gram = toks[i : i + L]
                if all(t in STOP for t in gram):
                    continue
                key = " ".join(gram)
                j = self.vocab.get(key)
                if j is not None and self.Ac.indptr[j + 1] - self.Ac.indptr[j] > 0:
                    found.append(j)
        return sorted(set(found))

    def ppr(self, seed, alpha=0.15, iters=30, tol=1e-8):
        """Power iteration for pi = alpha*s + (1-alpha)*pi P. Returns the full (n+m) vector."""
        s = seed / max(seed.sum(), 1e-12)
        pi = s.copy()
        for _ in range(iters):
            nxt = alpha * s + (1.0 - alpha) * (self.PT @ pi)
            if np.abs(nxt - pi).sum() < tol:
                pi = nxt
                break
            pi = nxt
        return pi

    def bridges(self, pids, top=3):
        """Entities shared between pairs of retrieved passages (used for evidence chains)."""
        out = {}
        sets = {p: set(self.A.getrow(p).indices) for p in pids}
        inv = {j: e for e, j in self.vocab.items()}
        for p in pids:
            shared = defaultdict(float)
            for q in pids:
                if q == p:
                    continue
                for j in sets[p] & sets[q]:
                    if self.A[p, j] > 0:
                        shared[j] += self.A[p, j]
            out[p] = [inv[j] for j, _ in sorted(shared.items(), key=lambda x: -x[1])[:top]]
        return out
