import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import lids.similarity as similarity_module
from lids import LIDS_similarity, get_text
from lids.direction_vectors import compute_direction_vectors


class FakeEmbedder:
    matrices = {
        "alpha": np.array([[2.0, 0.0], [0.0, 1.0]]),
        "beta": np.array([[1.0, 0.0], [0.0, 2.0]]),
    }

    def embed(self, text):
        return self.matrices[text]


class TestLIDS(unittest.TestCase):
    def test_public_signature_has_two_required_texts(self):
        parameters = list(inspect.signature(LIDS_similarity).parameters.values())
        required = [p.name for p in parameters if p.default is inspect.Parameter.empty]
        self.assertEqual(required, ["text1", "text2"])

    def test_get_text_reads_utf8(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_text("LIDS reads text files.\n", encoding="utf-8")
            self.assertEqual(get_text(path), "LIDS reads text files.\n")

    def test_similarity_uses_best_aligned_layer(self):
        expected = max(
            abs(similarity_module.cosine_similarity(a, b))
            for a, b in zip(
                compute_direction_vectors(FakeEmbedder.matrices["alpha"]),
                compute_direction_vectors(FakeEmbedder.matrices["beta"]),
            )
        )
        with patch.object(similarity_module, "_get_embedder", return_value=FakeEmbedder()):
            self.assertAlmostEqual(LIDS_similarity("alpha", "beta"), expected)

    def test_similarity_rejects_empty_text(self):
        with patch.object(similarity_module, "_get_embedder", return_value=FakeEmbedder()):
            for bad_text in ("", "   ", "\n\t"):
                with self.subTest(bad_text=bad_text), self.assertRaises(ValueError):
                    LIDS_similarity(bad_text, "alpha")

    def test_self_similarity_is_one(self):
        with patch.object(similarity_module, "_get_embedder", return_value=FakeEmbedder()):
            self.assertAlmostEqual(LIDS_similarity("alpha", "alpha"), 1.0)


if __name__ == "__main__":
    unittest.main()
