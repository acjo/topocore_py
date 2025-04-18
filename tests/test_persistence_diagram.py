from pathlib import Path

import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import the persistence computation functions from our implementation
from scipy.spatial.distance import cdist

from topocore.filtration import VRFiltration

# Import your custom implementations
from topocore.simplicial_complex import SimplicialComplex


def load_data(file_id):
    """Load the data from a CSV file.

    Parameters
    ----------
    file_id : int
        ID of the dataset (0-4 for files 1-5)

    Returns
    -------
    X : np.ndarray
        The point cloud data
    distance_matrix : np.ndarray
        The pairwise distance matrix
    """
    abs_path = Path(__file__).parent.parent
    data = pd.read_csv(
        abs_path / f"topocore/examples/data/CDHWdata_{file_id+1}.csv"
    )
    X = data.iloc[1:20, 1:].values
    distance_matrix = cdist(X, X, metric="euclidean")
    return X, distance_matrix


def compute_gudhi_persistence(
    points, max_dimension=3, max_edge_length=float("inf")
):
    """Compute persistent homology using GUDHI.

    Parameters
    ----------
    points : np.ndarray
        The point cloud data
    max_dimension : int
        Maximum homology dimension
    max_edge_length : float
        Maximum edge length for filtration

    Returns
    -------
    persistence_pairs : list
        List of (birth, death, dimension) tuples
    """
    # Create a Rips complex
    rips_complex = gd.RipsComplex(
        points=points, max_edge_length=max_edge_length
    )

    # Create the simplicial complex up to the specified dimension
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)

    # Compute the persistence diagram
    persistence = simplex_tree.persistence(
        homology_coeff_field=2, persistence_dim_max=True, min_persistence=-1
    )

    # Convert to the same format as our implementation
    persistence_pairs = [
        (birth, death, dim) for (dim, (birth, death)) in persistence
    ]

    return persistence_pairs, simplex_tree


def compare_persistence_results(custom_pairs, gudhi_pairs, tolerance=1e-6):
    """Compare the persistence pairs from custom implementation and GUDHI.

    Parameters
    ----------
    custom_pairs : list
        List of (birth, death, dimension) tuples from custom implementation
    gudhi_pairs : list
        List of (birth, death, dimension) tuples from GUDHI
    tolerance : float
        Tolerance for comparing float values

    Returns
    -------
    match_rate : float
        Percentage of matching pairs
    """
    # Sort both lists by dimension, birth, and death
    custom_pairs = sorted(custom_pairs, key=lambda x: (x[2], x[0], x[1]))
    gudhi_pairs = sorted(gudhi_pairs, key=lambda x: (x[2], x[0], x[1]))

    # Count matches
    matches = 0
    for i, (b1, d1, dim1) in enumerate(custom_pairs):
        if i < len(gudhi_pairs):
            b2, d2, dim2 = gudhi_pairs[i]
            if (
                dim1 == dim2
                and abs(b1 - b2) < tolerance
                and (
                    abs(d1 - d2) < tolerance
                    or (d1 == float("inf") and d2 == float("inf"))
                )
            ):
                matches += 1

    match_rate = matches / max(len(custom_pairs), len(gudhi_pairs)) * 100
    return match_rate


def plot_gudhi_persistence_diagram(simplex_tree):
    """Plot the persistence diagram using GUDHI's built-in plotting function.

    Parameters
    ----------
    simplex_tree : gudhi.SimplexTree
        The simplex tree with computed persistence
    """
    persistence = [(dim, (birth, death)) for birth, death, dim in simplex_tree]

    dims = [dim for dim, _ in persistence]
    print(max(dims))
    print(set(dims))
    return gd.plot_persistence_diagram(
        persistence=persistence, fontsize=15, alpha=0.4
    )


def test_persistence_implementation(file_id, max_dimension=3):
    """Test our persistence implementation against GUDHI.

    Parameters
    ----------
    file_id : int
        ID of the dataset (0-4 for files 1-5)
    max_dimension : int
        Maximum homology dimension
    """
    # Load data
    X, distance_matrix = load_data(file_id)

    print(f"Testing dataset {file_id+1}...")

    # Create filtration using our custom implementation
    filtration = VRFiltration(max_dimension=max_dimension, file_id=file_id)

    # Compute persistent homology using our implementation
    custom_persistence_pairs = filtration.compute_persistent_homology()

    # Compute persistent homology using GUDHI
    gudhi_persistence_pairs, simplex_tree = compute_gudhi_persistence(
        X, max_dimension
    )

    # Compare results
    match_rate = compare_persistence_results(
        custom_persistence_pairs, gudhi_persistence_pairs
    )
    print(f"Match rate: {match_rate:.2f}%")

    # Plot persistence diagrams
    filtration.plot_persistence_diagram(custom_persistence_pairs, max_dimension)
    plt.tight_layout()

    plot_gudhi_persistence_diagram(gudhi_persistence_pairs)
    plt.show()

    # Print detailed statistics
    print("\nDetailed statistics:")
    print(
        f"Custom implementation found {len(custom_persistence_pairs)} persistence pairs"
    )
    print(f"GUDHI found {len(gudhi_persistence_pairs)} persistence pairs")

    # Group by dimension
    custom_by_dim = {}
    gudhi_by_dim = {}

    for birth, death, dim in custom_persistence_pairs:
        if dim not in custom_by_dim:
            custom_by_dim[dim] = []
        custom_by_dim[dim].append((birth, death))

    for birth, death, dim in gudhi_persistence_pairs:
        if dim not in gudhi_by_dim:
            gudhi_by_dim[dim] = []
        gudhi_by_dim[dim].append((birth, death))

    # Print count per dimension
    print("\nPersistence pairs by dimension:")
    dims = sorted(set(list(custom_by_dim.keys()) + list(gudhi_by_dim.keys())))

    for dim in dims:
        custom_count = len(custom_by_dim.get(dim, []))
        gudhi_count = len(gudhi_by_dim.get(dim, []))
        print(f"H_{dim}: Custom={custom_count}, GUDHI={gudhi_count}")

    return custom_persistence_pairs, gudhi_persistence_pairs


def main():
    """Run tests on multiple datasets."""
    results = {}

    for file_id in range(5):
        try:
            custom_pairs, gudhi_pairs = test_persistence_implementation(file_id)
            results[file_id] = {
                "custom_pairs": custom_pairs,
                "gudhi_pairs": gudhi_pairs,
            }
            print("\n" + "=" * 50 + "\n")
        except Exception as e:
            print(f"Error testing dataset {file_id+1}: {str(e)}")

    return results


if __name__ == "__main__":
    main()
