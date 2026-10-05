import unittest

from kgrag.retrievers import Dense, Hybrid2
from kgrag.run import parse_spec

P = [
    {"title": "Alpha Town", "text": "Alpha Town is a town founded in 1900 by Brook Vale."},
    {"title": "Beta City", "text": "Beta City is a city founded in 1950 by Cole Dune."},
    {"title": "Brook Vale", "text": "Brook Vale was a settler born in Gamma Bay."},
    {"title": "Gamma Bay", "text": "Gamma Bay is a bay in the north."},
    {"title": "Delta", "text": "Delta is unrelated text about rivers and hills."},
]


class GateTest(unittest.TestCase):
    def setUp(self):
        self.d = Dense("tfidf").fit(P)

    def test_titles_gate_skips_two_named_titles(self):
        h = Hybrid2(self.d, gate="titles").fit(P)
        q = "Which was founded first, Alpha Town or Beta City?"
        pids, extra = h.search(q, 3)
        self.assertTrue(extra.get("gated"))
        self.assertEqual(pids, list(self.d.search(q, 3)[0]))

    def test_titles_gate_keeps_bridge_question(self):
        h = Hybrid2(self.d, gate="titles").fit(P)
        _, extra = h.search("Where was the founder of Alpha Town born?", 3)
        self.assertFalse(extra.get("gated"))

    def test_none_gate_never_skips(self):
        h = Hybrid2(self.d, gate="none").fit(P)
        self.assertFalse(h.search("Which was founded first, Alpha Town or Beta City?", 3)[1].get("gated"))

    def test_rule_and_oracle(self):
        self.assertTrue(Hybrid2(self.d, gate="rule").fit(P).search("Are both of them rivers?", 3)[1].get("gated"))
        o = Hybrid2(self.d, gate="oracle").fit(P)
        self.assertTrue(o.search("x?", 3, "comparison")[1].get("gated"))
        self.assertFalse(o.search("x?", 3, "bridge")[1].get("gated"))

    def test_parse_spec_strings(self):
        self.assertEqual(parse_spec("g:hybrid2:ner=both,m=3,gamma=0.3,tb=0.5,gate=titles")[2]["gate"], "titles")
        self.assertEqual(parse_spec("c:ce_h2:ner=both,m=3,top=20")[2]["top"], 20)


if __name__ == "__main__":
    unittest.main()
