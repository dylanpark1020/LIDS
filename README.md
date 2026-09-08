# LIDS similarity

LIDS (LLM Summary Inference Under the Layered Lens) is a semantic text-similarity metric based on BERT token embeddings and layered singular value decomposition (SVD). It compares the cumulative direction vectors of two texts across aligned latent layers and returns their maximum absolute cosine similarity.

Scores range from `0` to `1`; larger values indicate stronger semantic similarity.

## Paper

The methodology, theory, and empirical evaluation are described in:

> Dylan Park, Yingying Fan, and Jinchi Lv. [LIDS: LLM Summary Inference Under the Layered Lens](https://arxiv.org/abs/2603.00105). arXiv:2603.00105, 2026.

## Quick start

Requires Python 3.9 or newer. From the repository root:

```bash
python -m pip install -r requirements.txt
```

Then compare any two strings:

```python
from lids import LIDS_similarity

text1 = "Gardening requires regular attention to soil, sunlight, and watering."
text2 = "Healthy gardens depend on good soil, enough sun, and consistent watering."

score = LIDS_similarity(text1, text2)
print(f"LIDS similarity: {score:.4f}")
```

Or load plain-text files first:

```python
from lids import LIDS_similarity, get_text

text1 = get_text("examples/gardening_reference.txt")
text2 = get_text("examples/gardening_summary.txt")
score = LIDS_similarity(text1, text2)
```

The first call downloads and loads `bert-base-uncased`; later calls in the same Python process reuse the loaded model.

## Notebooks

- [`notebooks/LIDS_quickstart.ipynb`](notebooks/LIDS_quickstart.ipynb): install, load text, and calculate three example scores in a minute or two.
- [`notebooks/LIDS_demo.ipynb`](notebooks/LIDS_demo.ipynb): a deeper walkthrough of the paper's reference-to-summary BERT-SVD algorithm, optimal layer, and LIDS summary embedding.

Start Jupyter from the repository root so the relative example paths work:

```bash
jupyter notebook
```

## Public API

```python
LIDS_similarity(text1, text2, k=30, alpha=1.0,
                singular_value_fraction=1.0, max_length=512)
```

Only `text1` and `text2` are required. Both must be non-empty strings.

- `k` limits the number of aligned SVD layers.
- `alpha` controls singular value weighting.
- `singular_value_fraction` may stop at the smallest rank capturing the requested squared singular-value weight.
- `max_length` controls the BERT chunk length, including special tokens.

```python
get_text(file_path, encoding="utf-8")
```

Returns a text file's complete contents as a string.

## Method summary

For each text, LIDS:

1. Produces contextual token embeddings with `bert-base-uncased`.
2. Applies SVD to the token-embedding matrix.
3. Builds a cumulative direction vector at each retained SVD layer with sign alignment and singular-value weighting.
4. Returns the maximum absolute cosine similarity across aligned layers.

This repository implements the similarity-metric portion of the broader LIDS framework described in the accompanying research paper. The SOFARI visualization component is outside this package's current scope.

## Development

After installing the project, run the test suite:

```bash
python -m unittest discover -s tests
```

## License

This project is licensed under the terms in [`LICENSE`](LICENSE).
