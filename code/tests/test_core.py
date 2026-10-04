import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kgrag import metrics as M  # noqa: E402
from kgrag.text import extract_entities_regex, normalize_answer  # noqa: E402
from kgrag import data as D  # noqa: E402
from kgrag.retrievers import BM25, Dense, GraphPPR, Hybrid, Hybrid2  # noqa: E402


class TestMetrics(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize_answer("The Eiffel Tower!"), "eiffel tower")

    def test_em_f1(self):
        self.assertEqual(M.exact_match("the Paris", "Paris"), 1.0)
        self.assertAlmostEqual(M.f1("Paris France", "Paris"), 2 * 0.5 * 1 / 1.5)
        self.assertEqual(M.f1("Rome", "Paris"), 0.0)

    def test_recall(self):
        self.assertEqual(M.recall_at_k([1, 2, 3], {1, 9}), 0.5)
        self.assertEqual(M.all_support_at_k([1, 2, 9], {1, 9}), 1.0)

    def test_support(self):
        self.assertEqual(M.support_rate("Paris", "He was born in Paris."), 1.0)
        self.assertEqual(M.unsupported("Rome", "He was born in Paris."), 1.0)

    def test_bootstrap_detects_gap(self):
        a, b = [1.0] * 80 + [0.0] * 20, [0.0] * 60 + [1.0] * 40
        _, _, _, p = M.paired_bootstrap(a, b, n=2000)
        self.assertLess(p, 0.01)


class TestEntities(unittest.TestCase):
    def test_extract(self):
        ents = extract_entities_regex("The film was directed by Bob Smith in 1999 and shot in New York City.")
        self.assertIn("Bob Smith", ents)
        self.assertIn("1999", ents)
        self.assertIn("New York City", ents)


class TestRetrieval(unittest.TestCase):
    def test_toy_pipeline_recalls_bridge(self):
        exs, passages = D.load_toy(60, seed=1)
        dense = Dense("tfidf").fit(passages)
        hyb = Hybrid(dense, lam=0.5, beta=0.3).fit(passages)
        hits = 0
        for ex in exs:
            pids, _ = hyb.search(ex["question"], 5)
            hits += M.all_support_at_k(pids, ex["sup_pids"])
        self.assertGreater(hits / len(exs), 0.5)  # sanity only: the graph reaches the second hop

    def test_all_retrievers_return_k(self):
        exs, passages = D.load_toy(30, seed=2)
        dense = Dense("tfidf").fit(passages)
        for r in (BM25().fit(passages), dense, GraphPPR().fit(passages), Hybrid(dense).fit(passages)):
            pids, _ = r.search(exs[0]["question"], 5)
            self.assertLessEqual(len(pids), 5)
            self.assertGreater(len(pids), 0)



class TestHybrid2(unittest.TestCase):
    def test_runs_and_not_worse_than_chance(self):
        exs, passages = D.load("toy", None, 60, 0)
        dense = Dense("tfidf").fit(passages)
        for ner in ("regex", "both"):
            h = Hybrid2(dense, ner=ner).fit(passages)
            rec = []
            for e in exs:
                pids, extra = h.search(e["question"], 5)
                self.assertEqual(len(pids), 5)
                self.assertIn("bridges", extra)
                rec.append(M.all_support_at_k(pids, e["sup_pids"]))
            base = [M.all_support_at_k(dense.search(e["question"], 5)[0], e["sup_pids"]) for e in exs]
            self.assertGreaterEqual(sum(rec), sum(base) - 3)


if __name__ == "__main__":
    unittest.main()
