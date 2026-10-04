"""Retrievers: BM25, dense (TF-IDF or BGE), entity-graph PPR, and the proposed hybrid.

All retrievers expose fit(passages) and search(question, k) -> (list[pid], dict(extra)).
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .graph import EntityGraph


def _ptext(p):
    return p["title"] + ". " + p["text"]


def _minmax(x):
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


class BM25:
    name = "bm25"

    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b

    def fit(self, passages):
        self.cv = CountVectorizer(lowercase=True, token_pattern=r"\w+")
        X = self.cv.fit_transform([_ptext(p) for p in passages]).tocsc().astype(np.float64)
        n = X.shape[0]
        dl = np.asarray(X.sum(axis=1)).ravel()
        self.avg = dl.mean()
        df = np.asarray((X > 0).sum(axis=0)).ravel()
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        self.X, self.dl = X.tocsr(), dl
        return self

    def scores(self, q):
        qv = self.cv.transform([q])
        idx = qv.indices
        if len(idx) == 0:
            return np.zeros(self.X.shape[0])
        sub = self.X[:, idx].toarray()
        denom = sub + self.k1 * (1 - self.b + self.b * self.dl[:, None] / self.avg)
        return ((sub * (self.k1 + 1) / denom) * self.idf[idx]).sum(axis=1)

    def search(self, q, k):
        s = self.scores(q)
        return list(np.argsort(-s)[:k]), {}


class Dense:
    name = "dense"

    def __init__(self, backend="tfidf", model="BAAI/bge-small-en-v1.5", batch=64, cache=False):
        self.backend, self.model_name, self.batch = backend, model, batch
        self._cache = {} if cache else None

    def fit(self, passages):
        texts = [_ptext(p) for p in passages]
        if self.backend == "tfidf":
            self.vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1, stop_words="english")
            self.E = self.vec.fit_transform(texts)
        else:
            from sentence_transformers import SentenceTransformer

            self.st = SentenceTransformer(self.model_name)
            self.E = self.st.encode(texts, batch_size=self.batch, normalize_embeddings=True, show_progress_bar=False)
        return self

    def scores(self, q):
        if self._cache is not None and q in self._cache:
            return self._cache[q]
        if self.backend == "tfidf":
            out = np.asarray((self.E @ self.vec.transform([q]).T).todense()).ravel()
        else:
            qe = self.st.encode(
                ["Represent this sentence for searching relevant passages: " + q], normalize_embeddings=True
            )
            out = (self.E @ qe[0]).ravel()
        if self._cache is not None:
            self._cache[q] = out
        return out

    def search(self, q, k):
        s = self.scores(q)
        return list(np.argsort(-s)[:k]), {}


class GraphPPR:
    """Entity-graph retrieval seeded only by question entities (a GraphRAG-lite ablation)."""

    name = "graph"

    def __init__(self, alpha=0.15, ner="regex", idf=True, max_df_frac=0.2):
        self.alpha, self.ner, self.idf, self.max_df_frac = alpha, ner, idf, max_df_frac

    def fit(self, passages):
        self.g = EntityGraph(passages, self.ner, self.idf, self.max_df_frac)
        return self

    def search(self, q, k):
        ents = self.g.query_entities(q)
        if not ents:
            return [], {}
        seed = np.zeros(self.g.n + self.g.m)
        for j in ents:
            seed[self.g.n + j] = 1.0 / np.log(1.0 + self.g.df[j]) if self.idf else 1.0
        pi = self.g.ppr(seed, self.alpha)[: self.g.n]
        top = list(np.argsort(-pi)[:k])
        return top, {"bridges": self.g.bridges(top)}


class Hybrid:
    """Proposed method: dense-seeded + entity-seeded PPR, fused linearly with the dense score.

    seed = lam * (question-entity mass) + (1 - lam) * (top-s dense passages, softmax weighted)
    score = beta * minmax(dense) + (1 - beta) * minmax(PPR over passages)
    """

    name = "hybrid"

    def __init__(self, dense, alpha=0.15, lam=0.5, beta=0.5, seeds_k=5, ner="regex", idf=True, max_df_frac=0.2, temp=0.1):
        self.dense, self.alpha, self.lam, self.beta = dense, alpha, lam, beta
        self.seeds_k, self.ner, self.idf, self.max_df_frac, self.temp = seeds_k, ner, idf, max_df_frac, temp

    def fit(self, passages):
        self.g = EntityGraph(passages, self.ner, self.idf, self.max_df_frac)
        return self  # dense must already be fitted (shared across systems)

    def search(self, q, k):
        d = self.dense.scores(q)
        seed = np.zeros(self.g.n + self.g.m)
        top = np.argsort(-d)[: self.seeds_k]
        w = np.exp((d[top] - d[top].max()) / self.temp)
        seed[top] += (1 - self.lam) * w / w.sum()
        ents = self.g.query_entities(q)
        if ents and self.lam > 0:
            ew = np.array([1.0 / np.log(1.0 + self.g.df[j]) if self.idf else 1.0 for j in ents])
            seed[self.g.n + np.array(ents)] += self.lam * ew / ew.sum()
        pi = self.g.ppr(seed, self.alpha)[: self.g.n]
        score = self.beta * _minmax(d) + (1 - self.beta) * _minmax(pi)
        sel = list(np.argsort(-score)[:k])
        return sel, {"bridges": self.g.bridges(sel)}


class Hybrid2:
    """Proposed v2: graph-restricted two-hop re-ranking on top of a dense retriever.

    Stage 1  d = dense scores; first-hop set P1 = top-m passages by d.
    Stage 2  for each p in P1:
               bridge entities B(p) = entities of p that are not in the question and not hubs;
               candidates C(p)      = other passages that contain a bridge entity (graph neighbours through B(p));
               expanded query       = question + title(p) + first words of p;  s2_p(c) = dense score of the expanded query for c.
             boost(c) = max_p w_p * ( s2_p(c) + tb * [title(c) in B(p)] ),   w_p = softmax(d(p) / tau) normalised to max 1.
    Final    score(c) = dnorm(c) + gamma * boost(c).
    Passages outside every C(p) keep their dense score, so the graph can only re-order within linked evidence.
    """

    name = "hybrid2"

    def __init__(self, dense, m=3, gamma=1.0, tb=0.5, tau=0.1, exp_words=40, ner="regex", idf=True, max_df_frac=0.2):
        self.dense, self.m, self.gamma, self.tb, self.tau = dense, int(m), gamma, tb, tau
        self.exp_words, self.ner, self.idf, self.max_df_frac = exp_words, ner, idf, max_df_frac

    def fit(self, passages):
        from .text import ekey

        self.passages = passages
        self.g = EntityGraph(passages, self.ner, self.idf, self.max_df_frac)
        self.title_col = np.array([self.g.vocab.get(ekey(p["title"]), -1) for p in passages])
        self.inv = {j: e for e, j in self.g.vocab.items()}
        return self

    def _expand(self, q, pid):
        p = self.passages[pid]
        return q + " " + p["title"] + ". " + " ".join(p["text"].split()[: self.exp_words])

    def search(self, q, k):
        g = self.g
        d = self.dense.scores(q)
        lo, hi = float(d.min()), float(d.max())
        dn = (d - lo) / (hi - lo) if hi > lo else np.zeros_like(d)
        first = np.argsort(-d)[: self.m]
        w = np.exp((dn[first] - dn[first].max()) / self.tau)
        w = w / w.max()
        qset = set(g.query_entities(q))
        boost = np.zeros(g.n)
        for p, wp in zip(first, w):
            cols = g.A.indices[g.A.indptr[p] : g.A.indptr[p + 1]]
            bridge = [j for j in cols if j not in qset]
            if not bridge:
                continue
            bset = set(bridge)
            cand = np.unique(np.concatenate([g.Ac.indices[g.Ac.indptr[j] : g.Ac.indptr[j + 1]] for j in bridge]))
            cand = cand[cand != p]
            if len(cand) == 0:
                continue
            sc = self.dense.scores(self._expand(q, p))
            slo, shi = float(sc.min()), float(sc.max())
            s2 = (sc[cand] - slo) / (shi - slo) if shi > slo else np.zeros(len(cand))
            tl = np.array([self.title_col[c] in bset for c in cand], dtype=float)
            boost[cand] = np.maximum(boost[cand], wp * (s2 + self.tb * tl))
        score = dn + self.gamma * boost
        sel = list(np.argsort(-score)[:k])
        return sel, {"bridges": g.bridges(sel)}
