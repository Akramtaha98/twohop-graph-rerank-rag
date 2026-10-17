import unittest

import numpy as np

from kgrag.controls import components, variant_scores
from kgrag.retrievers import Dense, Hybrid2
from tests.test_gate import P


class ControlsTest(unittest.TestCase):
    def setUp(self):
        self.d = Dense("tfidf").fit(P)
        self.h = Hybrid2(self.d, m=2, gamma=0.3, tb=0.5, ner="regex").fit(P)

    def test_two_hop_matches_method(self):
        for q in ["Where was the founder of Alpha Town born?", "Which was founded first, Alpha Town or Beta City?", "What is Delta?"]:
            ref = self.h.search(q, 5)[0]
            _, dn, _, per = components(self.h, q)
            s = variant_scores(self.h, dn, per, 0.3, 0.5)["two_hop"]
            self.assertEqual(list(np.argsort(-s)[:5]), list(ref))

    def test_unlinked_keep_score_and_expand_all_touches_more(self):
        q = "Where was the founder of Alpha Town born?"
        _, dn, _, per = components(self.h, q)
        s = variant_scores(self.h, dn, per, 0.3, 0.5)
        linked = set()
        for p, wp, cand, tl, s2 in per:
            linked |= set(int(c) for c in cand)
        for i in range(len(P)):
            if i not in linked:
                self.assertAlmostEqual(s["two_hop"][i], dn[i])
                self.assertAlmostEqual(s["title_only"][i], dn[i])
        self.assertTrue(np.all(s["expand_all"] >= dn - 1e-12))
        self.assertTrue(np.all(s["title_only"] >= dn - 1e-12))

    def test_title_only_has_no_expansion_term(self):
        q = "Where was the founder of Alpha Town born?"
        _, dn, _, per = components(self.h, q)
        s = variant_scores(self.h, dn, per, 1.0, 0.5)["title_only"]
        self.assertTrue(np.all(s - dn <= 0.5 + 1e-9))


if __name__ == "__main__":
    unittest.main()


class ModeTest(unittest.TestCase):
    def test_modes_match_controls(self):
        d = Dense("tfidf").fit(P)
        for mode in ("two_hop", "title_only", "expand_all"):
            h = Hybrid2(d, m=2, gamma=0.7, tb=0.5, ner="regex", mode=mode).fit(P)
            for q in ["Where was the founder of Alpha Town born?", "Which was founded first, Alpha Town or Beta City?"]:
                _, dn, _, per = components(h, q)
                s = variant_scores(h, dn, per, 0.7, 0.5)[mode]
                self.assertEqual(list(np.argsort(-s)[:5]), list(h.search(q, 5)[0]), (mode, q))

    def test_title_only_makes_no_extra_encodings(self):
        class Counting(Dense):
            calls = 0

            def scores(self, q):
                Counting.calls += 1
                return super().scores(q)

        d = Counting("tfidf").fit(P)
        h = Hybrid2(d, m=3, gamma=1.0, tb=0.5, ner="regex", mode="title_only").fit(P)
        h.search("Where was the founder of Alpha Town born?", 3)
        self.assertEqual(Counting.calls, 1)
