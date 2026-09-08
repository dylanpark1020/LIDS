import numpy as np


DEFAULT_K = 30
DEFAULT_ALPHA = 1.0
DEFAULT_SINGULAR_VALUE_FRACTION = 1.0


def percent_weight_sqr_sing_vals(fraction, singular_values):
    """
    Find the smallest number of singular values whose squared sum
    contains the specified fraction of the total squared singular values.

    Parameters
    ----------
    fraction : float
        Desired fraction of total squared singular-value weight.

    singular_values : numpy.ndarray
        Singular values from an SVD.

    Returns
    -------
    int
        Number of singular values required.
    """

    if not 0 < fraction <= 1:
        raise ValueError("fraction must satisfy 0 < fraction <= 1.")

    total = np.sum(singular_values ** 2)

    cutoff = total * fraction

    cumulative = 0.0

    for count, singular_value in enumerate(singular_values, start=1):
        cumulative += singular_value ** 2

        if cumulative >= cutoff:
            return count

    return len(singular_values)


def compute_direction_vectors(
    embedding_matrix,
    k=DEFAULT_K,
    alpha=DEFAULT_ALPHA,
    singular_value_fraction=DEFAULT_SINGULAR_VALUE_FRACTION,
):
    """
    Construct cumulative LIDS direction vectors from an embedding matrix.

    The procedure is:

        embedding matrix
            ↓
           SVD
            ↓
        rank truncation
            ↓
        singular-value weighting
            ↓
        direction vector

    Parameters
    ----------
    embedding_matrix : numpy.ndarray
        Token embedding matrix Y.

    k : int, optional
        Maximum number of singular directions to use.

    alpha : float, optional
        Singular-value weighting exponent.

    singular_value_fraction : float, optional
        Fraction of squared singular-value weight used to determine
        the effective rank.

    Returns
    -------
    list[numpy.ndarray]
        Direction vector for each retained layer, from ``d(1)`` through
        ``d(effective_rank)``.
    """

    embedding_matrix = np.asarray(embedding_matrix, dtype=float)

    if embedding_matrix.ndim != 2:
        raise ValueError("embedding_matrix must be a two-dimensional array.")

    if embedding_matrix.shape[0] == 0:
        raise ValueError("embedding_matrix must contain at least one token.")

    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be at least 1.")

    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be a positive finite number.")

    # --------------------------------------------------------
    # SVD
    # --------------------------------------------------------

    U, singular_values, Vt = np.linalg.svd(
        embedding_matrix,
        full_matrices=False,
    )

    # --------------------------------------------------------
    # Determine effective rank
    # --------------------------------------------------------

    effective_rank = percent_weight_sqr_sing_vals(
        singular_value_fraction,
        singular_values,
    )

    effective_rank = min(
        effective_rank,
        k,
        len(singular_values),
    )

    U_k = U[:, :effective_rank]
    singular_values_k = singular_values[:effective_rank]

    # --------------------------------------------------------
    # Resolve SVD sign ambiguity
    #
    # The sign of singular vectors is arbitrary. We orient each
    # singular vector consistently using the all-ones vector.
    # --------------------------------------------------------

    V_k = Vt[:effective_rank].T
    ones_normalized = np.ones(V_k.shape[0]) / np.sqrt(V_k.shape[0])
    direction = np.zeros(embedding_matrix.shape[1], dtype=float)
    directions = []

    for layer in range(effective_rank):
        sign = np.sign(np.dot(V_k[:, layer], ones_normalized))
        if sign == 0:
            sign = 1.0

        direction += (
            singular_values_k[layer] ** alpha
            * sign
            * (embedding_matrix.T @ U_k[:, layer])
        )
        directions.append(direction.copy())

    return directions


def compute_direction_vector(
    embedding_matrix,
    k=DEFAULT_K,
    alpha=DEFAULT_ALPHA,
    singular_value_fraction=DEFAULT_SINGULAR_VALUE_FRACTION,
):
    """Return the final cumulative vector (backward-compatible helper)."""
    return compute_direction_vectors(
        embedding_matrix,
        k=k,
        alpha=alpha,
        singular_value_fraction=singular_value_fraction,
    )[-1]
