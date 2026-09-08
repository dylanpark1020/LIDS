import numpy as np

from .direction_vectors import compute_direction_vectors


DEFAULT_ALPHA = 1.0
DEFAULT_K = 30
DEFAULT_SINGULAR_VALUE_FRACTION = 1.0
DEFAULT_MAX_LENGTH = 512


# Load BERT once rather than loading it every time
# LIDS_similarity is called.
_embedders = {}


def _get_embedder(max_length=DEFAULT_MAX_LENGTH):
    if max_length not in _embedders:
        from .embeddings import BERTEmbedder

        _embedders[max_length] = BERTEmbedder(max_length=max_length)

    return _embedders[max_length]


def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    """

    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError(
            "Cannot compute cosine similarity with a zero vector."
        )

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def LIDS_similarity(
    text1,
    text2,
    k=DEFAULT_K,
    alpha=DEFAULT_ALPHA,
    singular_value_fraction=DEFAULT_SINGULAR_VALUE_FRACTION,
    max_length=DEFAULT_MAX_LENGTH,
):
    """
    Compute the LIDS similarity between two texts.

    Parameters
    ----------
    text1 : str
        First text.

    text2 : str
        Second text.

    k : int, optional
        Maximum number of aligned SVD layers to compare.

    alpha : float, optional
        Singular value weighting exponent.

    singular_value_fraction : float, optional
        Fraction of squared singular value weight used when
        determining the effective rank.

    max_length : int, optional
        Maximum BERT sequence length.

    Returns
    -------
    float
        LIDS similarity score.
    """

    if not isinstance(text1, str):
        raise TypeError("text1 must be a string.")

    if not isinstance(text2, str):
        raise TypeError("text2 must be a string.")

    if not text1.strip():
        raise ValueError("text1 must contain at least one non-whitespace character.")

    if not text2.strip():
        raise ValueError("text2 must contain at least one non-whitespace character.")

    embedder = _get_embedder(max_length=max_length)

    # --------------------------------------------------------
    # Convert both texts into BERT embedding matrices
    # --------------------------------------------------------

    matrix1 = embedder.embed(text1)
    matrix2 = embedder.embed(text2)

    # --------------------------------------------------------
    # Construct LIDS direction vectors
    # --------------------------------------------------------

    direction_vectors1 = compute_direction_vectors(
        matrix1,
        k=k,
        alpha=alpha,
        singular_value_fraction=singular_value_fraction,
    )

    direction_vectors2 = compute_direction_vectors(
        matrix2,
        k=k,
        alpha=alpha,
        singular_value_fraction=singular_value_fraction,
    )

    # --------------------------------------------------------
    # Compare the two direction vectors
    # --------------------------------------------------------

    aligned_layers = min(len(direction_vectors1), len(direction_vectors2))
    similarities = [
        abs(cosine_similarity(direction_vectors1[i], direction_vectors2[i]))
        for i in range(aligned_layers)
    ]

    # Guard against tiny floating-point excursions outside the theoretical range.
    return float(np.clip(max(similarities), 0.0, 1.0))
