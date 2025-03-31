"""Linear Algebra subroutines for Z mod 2."""

import numpy as np
from numba import njit


@njit
def rref_mod2(A: np.ndarray) -> tuple[np.ndarray, list]:
    """Compute the row reduced echelon form of matrix A over Z mod 2.

    Parameters
    ----------
    A : ndarray (m,n)
        Matrix to find the rref form of.

    Returns
    -------
    A : ndarray(m,n)
        A in row-reduced echelon form
    pivot_cols : list
        List of indices of the pivot columns

    Raises
    ------
    ValueError
        If matrix A is not in mod 2 form.
    """
    if np.any(A % 2 != A):
        raise ValueError("Matrix A is not in mod 2 form.")

    A = A.copy().astype(np.int64)
    m, n = A.shape

    # Keep track of pivot columns
    pivot_cols = []

    i = 0  # row index
    for j in range(n):  # iterate through columns
        # Find pivot in column j, starting from row i
        pivot_row: int = -1
        for k in range(i, m):
            if A[k, j] == 1:
                pivot_row = k
                break

        if pivot_row != -1:
            # Remember this as a pivot column
            pivot_cols.append(j)

            # Swap rows if needed
            if pivot_row != i:
                temp = A[i].copy()
                A[i] = A[pivot_row]
                A[pivot_row] = temp

            # Eliminate other rows
            for k in range(m):
                if k != i and A[k, j] == 1:
                    A[k] = (A[k] + A[i]) % 2

            i += 1
            if i == m:  # No more rows
                break

    return A, pivot_cols


@njit
def nullspace_mod2(A: np.ndarray) -> list[np.ndarray]:
    """Find a basis for the nullspace of matrix A over Z mod 2.

    Parameters
    ----------
    A : (m,n) ndarray
        Matrix to find the null space of

    Returns
    -------
    basis : list of numpy arrays
        List containing the basis vectors for the nullspace in mod2.

    Raises
    ------
    ValueError
        If matrix A is not in mod 2 form.
    """
    if np.any(A % 2 != A):
        raise ValueError("Matrix A is not in mod 2 form.")
    rref, pivot_cols = rref_mod2(A)

    m, n = A.shape

    # Free columns are those that aren't pivot columns
    free_cols = [j for j in range(n) if j not in pivot_cols]

    # Create basis vectors
    basis = []
    for free_col in free_cols:
        # Create a vector to solve Ax = 0
        v = np.zeros(n, dtype=np.int64)
        v[free_col] = 1

        # Set the values for pivot variables
        for i, pivot_col in enumerate(pivot_cols):
            if i < rref.shape[0]:  # Check if we have this many pivot rows
                v[pivot_col] = rref[i, free_col]

        # In Z mod 2, we need to negate the values in pivot positions
        for i, pivot_col in enumerate(pivot_cols):
            if i < rref.shape[0]:
                v[pivot_col] = (-v[pivot_col]) % 2

        basis.append(v)

    return basis


@njit
def image_mod2(A: np.ndarray) -> list[np.ndarray]:
    """Find a basis for the image of the matrix.

    (column space) of matrix A over Z mod 2

    Parameters
    ----------
    A : (m,n) numpy array
        Matrix to find basis of the image of.

    Returns
    -------
    basis : list of numpy arrays
        List of basis vectors

    Raises
    ------
    ValueError
        If matrix A is not in mod 2 form.
    """
    if np.any(A % 2 != A):
        raise ValueError("Matrix A is not in mod 2 form.")

    # Compute RREF and find pivot columns
    rref, pivot_cols = rref_mod2(A)

    # The pivot columns of the original matrix form a basis for the image
    basis = [A[:, j] for j in pivot_cols]

    return basis
