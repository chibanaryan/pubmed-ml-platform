"""Tests for the evaluation harness."""

from src.embeddings.evaluate import compute_relevance_score, dcg, ndcg


class TestRelevanceScoring:
    def test_high_relevance_two_hits(self):
        mesh = {"Creatine", "Dietary Supplements", "Humans"}
        query = {
            "high_relevance_mesh": ["Creatine", "Dietary Supplements"],
            "medium_relevance_mesh": ["Muscle, Skeletal"],
            "low_relevance_mesh": ["Sports Nutritional Physiological Phenomena"],
        }
        assert compute_relevance_score(mesh, query) == 3

    def test_high_relevance_one_hit_plus_medium(self):
        mesh = {"Creatine", "Muscle, Skeletal"}
        query = {
            "high_relevance_mesh": ["Creatine", "Dietary Supplements"],
            "medium_relevance_mesh": ["Muscle, Skeletal"],
            "low_relevance_mesh": [],
        }
        assert compute_relevance_score(mesh, query) == 3

    def test_high_relevance_one_hit_no_medium(self):
        mesh = {"Creatine", "Humans"}
        query = {
            "high_relevance_mesh": ["Creatine", "Dietary Supplements"],
            "medium_relevance_mesh": ["Muscle, Skeletal"],
            "low_relevance_mesh": [],
        }
        assert compute_relevance_score(mesh, query) == 2

    def test_medium_relevance(self):
        mesh = {"Muscle, Skeletal", "Exercise"}
        query = {
            "high_relevance_mesh": ["Creatine"],
            "medium_relevance_mesh": ["Muscle, Skeletal", "Exercise"],
            "low_relevance_mesh": [],
        }
        assert compute_relevance_score(mesh, query) == 2

    def test_low_relevance(self):
        mesh = {"Humans", "Exercise"}
        query = {
            "high_relevance_mesh": ["Creatine"],
            "medium_relevance_mesh": ["Muscle, Skeletal", "Exercise"],
            "low_relevance_mesh": [],
        }
        assert compute_relevance_score(mesh, query) == 1

    def test_no_relevance(self):
        mesh = {"Humans", "Male", "Female"}
        query = {
            "high_relevance_mesh": ["Creatine"],
            "medium_relevance_mesh": ["Muscle, Skeletal"],
            "low_relevance_mesh": ["Exercise"],
        }
        assert compute_relevance_score(mesh, query) == 0


class TestNDCG:
    def test_perfect_ranking(self):
        # All relevant at top
        rels = [3, 3, 2, 1, 0]
        assert ndcg(rels, 5) == 1.0

    def test_worst_ranking(self):
        # All irrelevant
        rels = [0, 0, 0, 0, 0]
        assert ndcg(rels, 5) == 0.0

    def test_reversed_ranking(self):
        # Relevant items at bottom
        rels = [0, 0, 0, 3, 3]
        assert 0 < ndcg(rels, 5) < 1.0

    def test_dcg_values(self):
        # DCG = 3/log2(2) + 2/log2(3) + 1/log2(4)
        rels = [3, 2, 1]
        expected = 3 / 1.0 + 2 / 1.585 + 1 / 2.0
        assert abs(dcg(rels, 3) - expected) < 0.01

    def test_ndcg_k_truncation(self):
        rels = [3, 2, 1, 0, 0, 0, 0, 0, 0, 0]
        assert ndcg(rels, 3) == ndcg([3, 2, 1], 3)
