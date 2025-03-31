import numpy as np
from abstract_simplical_complex import rref_mod2


def test_rref_identity():
    A = np.eye(3, dtype=int)

    R, pivot_cols = rref_mod2(A)

    assert np.all(R == A), "RREF is incorrect for identity"

    assert sorted(pivot_cols) == sorted(
        [0, 1, 2]
    ), "Pivot column indices are incorrect."


def test_rref_ones():
    A = np.ones((3, 3), dtype=int)

    R, pivot_cols = rref_mod2(A)

    correct = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=int)

    assert np.all(R == correct), "RREF is incorrect for identity"

    assert pivot_cols == [0], "Pivot column indices are incorrect."


def test_rref_non_trivial():
    A = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0]])

    R, pivot_cols = rref_mod2(A)

    correct = np.array(
        [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0]], dtype=int
    )
    cols = [0, 1, 2]

    assert np.all(R == correct), "Non trivial case fails."
    assert cols == pivot_cols, "Pivot cols fail."
