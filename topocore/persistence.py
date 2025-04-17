"""Persistent Homology Implementation."""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from topocore.linalg import rref_mod2
from topocore.topocore import SimplicialComplex


def compute_persistence_pairs(boundary_matrices):
    """Compute persistence pairs from a list of boundary matrices.

    Parameters
    ----------
    boundary_matrices : list of np.ndarray
        List of boundary matrices [∂₀, ∂₁, ∂₂, ...]

    Returns
    -------
    pairs : dict
        Dictionary mapping dimension to list of (birth, death) pairs
    """
    max_dim = len(boundary_matrices) - 1
    pairs = defaultdict(list)

    # Track the lowest non-zero entry for each column
    lowest = {}  # Maps (dim, col_idx) to row_idx

    # First, compute the lowest non-zero entry for each column in each dimension
    for dim in range(max_dim + 1):
        matrix = boundary_matrices[dim]
        _, n = matrix.shape

        for j in range(n):
            col = matrix[:, j]
            non_zero = np.where(col == 1)[0]

            if len(non_zero) > 0:
                pivot = np.max(non_zero)
                lowest[(dim, j)] = pivot

    # Now compute persistence pairs
    for dim in range(max_dim):
        matrix = boundary_matrices[dim]
        next_matrix = boundary_matrices[dim + 1]

        m, n = matrix.shape
        _, n_next = next_matrix.shape

        # Set to track which columns are paired
        paired_birth = set()
        paired_death = set()

        # Process each column in the next dimension
        for j in range(n_next):
            col = next_matrix[:, j].copy()

            # Reduce the column
            while True:
                non_zero = np.where(col == 1)[0]
                if len(non_zero) == 0:
                    break

                pivot = np.max(non_zero)

                # Find a column with the same pivot
                pivot_col = None
                for k in range(n):
                    if (
                        (dim, k) in lowest
                        and lowest[(dim, k)] == pivot
                        and k not in paired_birth
                    ):
                        pivot_col = k
                        break

                if pivot_col is None:
                    break

                # Add column pivot_col to column j (this is where we need to be careful)
                # Both columns reference the same matrix, so they have compatible shapes
                reducer_col = matrix[:, pivot_col]
                col = np.mod(col + reducer_col, 2)

            # After reduction, check if this column creates a persistence pair
            non_zero = np.where(col == 1)[0]
            if len(non_zero) > 0:
                pivot = np.max(non_zero)

                # Find a column with this pivot
                for k in range(n):
                    if (
                        (dim, k) in lowest
                        and lowest[(dim, k)] == pivot
                        and k not in paired_birth
                    ):
                        # Create persistence pair
                        pairs[dim].append((k, j))
                        paired_birth.add(k)
                        paired_death.add(j)
                        break

            # Update the lowest entry for this column
            if len(non_zero) > 0:
                lowest[(dim + 1, j)] = np.max(non_zero)

        # Essential classes in dimension dim
        for j in range(n):
            if (dim, j) in lowest and j not in paired_birth:
                pairs[dim].append((j, float("inf")))

    # Essential classes in the highest dimension
    for j in range(boundary_matrices[max_dim].shape[1]):
        if (max_dim, j) in lowest and j not in paired_death:
            pairs[max_dim].append((j, float("inf")))

    return pairs


def compute_persistence_diagram(
    filtration_complexes: list[SimplicialComplex], max_dimension: int = 2
):
    """Compute the persistence diagram for a filtration.

    Parameters
    ----------
    filtration_complexes : list of SimplicialComplex
        List of simplicial complexes forming a filtration
    max_dimension : int
        Maximum dimension to compute persistence for

    Returns
    -------
    diagram : dict
        Dictionary mapping dimension to list of (birth, death, death_val - birth_val) tuples
    """
    # Extract threshold values
    threshold_values = [
        complex.filtration_value for complex in filtration_complexes
    ]

    # Use the final complex for the boundary matrices
    final_complex = filtration_complexes[-1]
    final_complex.set_simplices_as_lists()

    # Construct boundary matrices
    boundary_matrices = []
    for dim in range(max_dimension + 1):
        if dim in final_complex.simplices_list:
            boundary_matrices.append(final_complex.compute_boundary_matrix(dim))
        else:
            # Create an empty matrix if no simplices of this dimension
            if dim == 0:
                # For dimension 0, shape is (1, number of vertices)
                n_vertices = len(final_complex.simplices_list.get(0, []))
                boundary_matrices.append(np.zeros((1, n_vertices), dtype=int))
            else:
                # For other dimensions, shape depends on previous dimension
                prev_dim_count = len(
                    final_complex.simplices_list.get(dim - 1, [])
                )
                boundary_matrices.append(
                    np.zeros((prev_dim_count, 0), dtype=int)
                )

    # Compute persistence pairs
    pairs = compute_persistence_pairs(boundary_matrices)

    # Map simplex indices to filtration indices
    simplex_to_filtration = {}
    for dim in range(max_dimension + 1):
        simplices = final_complex.simplices_list.get(dim, [])
        for i, simplex in enumerate(simplices):
            # Find the earliest filtration index where this simplex appears
            for fi, complex in enumerate(filtration_complexes):
                if simplex in complex.simplices.get(dim, set()):
                    simplex_to_filtration[(dim, i)] = fi
                    break

    # Convert to persistence diagram format
    diagram = defaultdict(list)

    for dim in range(max_dimension + 1):
        for birth_idx, death_idx in pairs[dim]:
            if birth_idx[0] == float("inf"):
                # Skip invalid pairs
                continue

            # Get birth filtration index
            birth_fi = simplex_to_filtration.get(birth_idx, 0)
            birth_val = threshold_values[birth_fi]

            # Get death filtration index
            if death_idx[0] == float("inf"):
                death_val = float("inf")
            else:
                death_fi = simplex_to_filtration.get(
                    death_idx, len(threshold_values) - 1
                )
                death_val = threshold_values[death_fi]

            # Calculate persistence
            persistence = (
                float("inf")
                if death_val == float("inf")
                else death_val - birth_val
            )

            if persistence > 0:  # Only include positive persistence
                diagram[dim].append((birth_val, death_val, persistence))

    return diagram


def plot_persistence_diagram(
    diagram: dict, max_dimension: int = 2, title: str = "Persistence Diagram"
):
    """Plot the persistence diagram.

    Parameters
    ----------
    diagram : dict
        Dictionary mapping dimension to list of (birth, death, persistence) tuples
    max_dimension : int
        Maximum dimension to include in the diagram
    title : str
        Title for the plot
    """
    plt.figure(figsize=(10, 8))

    colors = ["b", "r", "g", "c", "m", "y", "k"]
    markers = ["o", "s", "D", "^", "v", "p", "*"]

    # Check if diagram is empty
    is_empty = True
    for dim in range(max_dimension + 1):
        if dim in diagram and diagram[dim]:
            is_empty = False
            break

    if is_empty:
        plt.text(
            0.5,
            0.5,
            "No persistent homology features found",
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.title(title)
        return plt

    # Find max value for plot bounds
    max_val = 0
    for dim in range(max_dimension + 1):
        if dim in diagram and diagram[dim]:
            for birth, death, _ in diagram[dim]:
                if birth > max_val:
                    max_val = birth
                if death != float("inf") and death > max_val:
                    max_val = death

    # If still no valid max_val, set a default
    if max_val == 0:
        max_val = 1.0

    for dim in range(max_dimension + 1):
        if dim not in diagram or not diagram[dim]:
            continue

        points = diagram[dim]

        births = [p[0] for p in points]
        deaths = [
            p[1] if p[1] != float("inf") else max_val * 1.2 for p in points
        ]

        plt.scatter(
            births,
            deaths,
            color=colors[dim % len(colors)],
            marker=markers[dim % len(markers)],
            label=f"H{dim}",
            alpha=0.7,
        )

    # Plot diagonal
    diag_min = 0
    diag_max = max_val * 1.2
    plt.plot([diag_min, diag_max], [diag_min, diag_max], "k--", alpha=0.5)

    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return plt
