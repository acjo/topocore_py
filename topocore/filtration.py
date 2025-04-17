"""Filtration code."""

from itertools import chain
from pathlib import Path
from types import NoneType
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams
from scipy.spatial.distance import cdist
from tqdm import tqdm

from topocore.linalg import image_mod2, nullspace_mod2, rref_mod2
from topocore.persistence import (
    compute_persistence_diagram,
    compute_persistence_pairs,
    plot_persistence_diagram,
)
from topocore.topocore import SimplicialComplex

rcParams["font.family"] = "serif"
rcParams["font.size"] = 15
rcParams["axes.labelsize"] = 10
rcParams["xtick.labelsize"] = 10
rcParams["ytick.labelsize"] = 10
rcParams["legend.fontsize"] = 10
rcParams["figure.figsize"] = (7, 7)
rcParams["figure.dpi"] = 200
rcParams["axes.titlesize"] = 15


class VRFiltration(object):
    """Vietoris-Rips Filtration.

    Notes
    -----
    Building the full filtration is incredibly computationally complex.
    So we include a sub_sample parameter that says to include include a certain number of threshold values

    Parmeters
    ---------
    max_dimension : int
        The max dimensional simplices to include in the complexes of the filtration. Default, 3
    file_id : int
        the id of the file to read in. Default, 0
    sub_sample : int, Optional
        How many threshold values to include. Defualt, 20.
        If None, there will be no sampling.
    """

    def __init__(
        self,
        max_dimension: int = 3,
        file_id: int = 0,
        sub_sample: Optional[int] = None,
    ) -> None:
        self.max_dimension = max_dimension
        self.file_id = file_id
        self.sub_sample = sub_sample

        file_path = Path(__file__)
        data_path = file_path.parent
        data = pd.read_csv(
            data_path / f"examples/data/CDHWdata_{file_id+1}.csv"
        )

        X = data.iloc[:, 1:].values
        distance_matrix = cdist(X, X, metric="euclidean")
        n = distance_matrix.shape[0]
        # Get unique distance values (excluding zeros on diagonal)
        unique_distances = np.unique(distance_matrix[distance_matrix > 0])
        filtration_values = np.sort(unique_distances)

        self.filtration_complexes: list[SimplicialComplex] = (
            SimplicialComplex.build_filtration_incrementally(
                distance_matrix, filtration_values[:sub_sample], max_dimension
            )
        )

        return

    def compute_persistent_homology(self, max_dim: Optional[int] = None):
        """Compute persistent homology for the filtration.

        Parameters
        ----------
        max_dim : int, Optional
            Maximum dimension to compute homology for. If None, uses self.max_dimension

        Returns
        -------
        diagram : dict
            Persistence diagram
        """
        if max_dim is None:
            max_dim = self.max_dimension

        diagram = compute_persistence_diagram(
            self.filtration_complexes, max_dim
        )
        return diagram

    def plot_persistence_diagram(
        self, max_dim: Optional[int] = None, title: Optional[str] = None
    ):
        """Plot the persistence diagram.

        Parameters
        ----------
        max_dim : int, optional
            Maximum dimension to include in the diagram
        title : str, optional
            Title for the plot
        """
        if max_dim is None:
            max_dim = self.max_dimension

        if title is None:
            title = f"Persistence Diagram (Data File {self.file_id+1})"

        diagram = self.compute_persistent_homology(max_dim)
        plt_obj = plot_persistence_diagram(diagram, max_dim, title)

        return plt_obj

    def plot_barcode(
        self, max_dim: Optional[int] = None, title: Optional[str] = None
    ):
        """Plot the barcode representation of the persistence diagram.

        Parameters
        ----------
        max_dim : int, optional
            Maximum dimension to include in the barcode
        title : str, optional
            Title for the plot
        """
        if max_dim is None:
            max_dim = self.max_dimension

        if title is None:
            title = f"Persistence Barcode (Data File {self.file_id+1})"

        diagram = self.compute_persistent_homology(max_dim)

        plt.figure(figsize=(10, 8))

        colors = ["b", "r", "g", "c", "m", "y", "k"]

        # Count total bars for vertical spacing
        total_bars = sum(
            len(diagram.get(dim, [])) for dim in range(max_dim + 1)
        )

        if total_bars == 0:
            plt.text(
                0.5,
                0.5,
                "No persistent homology features found",
                horizontalalignment="center",
                verticalalignment="center",
            )
            plt.title(title)
            return plt

        current_bar = 0

        # Find the maximum finite death value for plot limits
        max_death = 0
        for dim in range(max_dim + 1):
            if dim not in diagram:
                continue
            for _, death, _ in diagram[dim]:
                if death != float("inf") and death > max_death:
                    max_death = death

        y_ticks = []
        y_labels = []

        for dim in range(max_dim + 1):
            if dim not in diagram:
                continue

            # Sort by persistence for better visualization
            points = sorted(diagram[dim], key=lambda x: x[2], reverse=True)

            for birth, death, _ in points:
                # Plot a horizontal line for each bar
                if death == float("inf"):
                    plt.plot(
                        [birth, max_death * 1.2],
                        [
                            total_bars - current_bar,
                            total_bars - current_bar,
                        ],
                        colors[dim % len(colors)],
                        linewidth=2,
                    )
                    plt.plot(
                        [
                            max_death * 1.2,
                            max_death * 1.2 + max_death * 0.1,
                        ],
                        [
                            total_bars - current_bar,
                            total_bars - current_bar,
                        ],
                        colors[dim % len(colors)],
                        linewidth=2,
                        linestyle="--",
                    )
                else:
                    plt.plot(
                        [birth, death],
                        [
                            total_bars - current_bar,
                            total_bars - current_bar,
                        ],
                        colors[dim % len(colors)],
                        linewidth=2,
                    )

                y_ticks.append(total_bars - current_bar)
                y_labels.append(f"H{dim}")

                current_bar += 1

        plt.yticks(y_ticks, y_labels)
        plt.xlabel("Threshold Value")
        plt.title(title)
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        return plt

    def analyze_persistence(
        self,
        max_dim: Optional[int] = None,
        persistence_threshold: Optional[float] = None,
    ):
        """Analyze the persistence diagram to identify significant features.

        Parameters
        ----------
        max_dim : int, optional
            Maximum dimension to analyze
        persistence_threshold : float, optional
            Threshold for considering a feature significant. If None, uses the 75th percentile.

        Returns
        -------
        significant_features : dict
            Dictionary with analysis of significant features
        """
        if max_dim is None:
            max_dim = self.max_dimension

        diagram = self.compute_persistent_homology(max_dim)

        # Collect all finite persistence values to determine threshold
        all_persistence = []
        for dim in range(max_dim + 1):
            if dim not in diagram:
                continue
            for _, death, persistence in diagram[dim]:
                if persistence != float("inf"):
                    all_persistence.append(persistence)

        if not all_persistence:
            return {
                "message": "No finite persistence features found",
                "features": {},
            }

        # Set threshold to 75th percentile if not provided
        if persistence_threshold is None:
            persistence_threshold = float(np.percentile(all_persistence, 75))

        significant_features = {
            "threshold": persistence_threshold,
            "features": {},
        }

        for dim in range(max_dim + 1):
            if dim not in diagram:
                continue

            # Sort by persistence
            persistent_features = sorted(
                diagram[dim], key=lambda x: x[2], reverse=True
            )

            # Filter significant features
            significant = [
                f
                for f in persistent_features
                if f[2] > persistence_threshold or f[2] == float("inf")
            ]

            if significant:
                significant_features["features"][dim] = {
                    "count": len(significant),
                    "details": significant,
                }

        return significant_features

    def compute_euler_characteristic(self):
        """Compute the Euler characteristic for each complex in the filtration.

        Returns
        -------
        euler_characteristics : list of int
            Euler characteristic for each threshold value
        """
        euler_characteristics = []

        for complex in self.filtration_complexes:
            # Set the simplices_list attribute if not already set
            if not hasattr(complex, "simplices_list"):
                complex.set_simplices_as_lists()

            # Compute Euler characteristic
            euler = sum(
                (-1) ** dim * len(simplices)
                for dim, simplices in complex.simplices.items()
            )
            euler_characteristics.append(euler)

        return euler_characteristics
