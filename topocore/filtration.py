"""Filtration code."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams
from scipy.spatial.distance import cdist

from topocore.simplicial_complex import SimplicialComplex
from topocore.persistence import (reduce_boundary_matrix, 
                                  build_persistence_boundary_matrix, 
                                  extract_persistence_pairs)

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

    def compute_persistent_homology(self):
        """Compute persistent homology for a filtration.
        
        Parameters
        ----------
        filtration : VRFiltration
            Vietoris-Rips filtration object
        
        Returns
        -------
        persistence_pairs : list
            A list of (birth, death, dimension) tuples representing the persistence diagram
        """
        # Build the boundary matrix for the persistence algorithm
        print("Buidling boundary matrix...")
        D, simplex_info = build_persistence_boundary_matrix(self)
        print("Complete!")
        
        # Reduce the boundary matrix
        print("Reducing boundary matrix...")
        print(D.shape)
        R, low = reduce_boundary_matrix(D)
        print("Complete!")
        
        # Extract the persistence pairs
        print("Extracting persistence pairs...")
        persistence_pairs = extract_persistence_pairs(R, low, simplex_info, self)
        print("Complete!")
        
        return persistence_pairs

    def plot_persistence_diagram(self, persistence_pairs:list, max_dimension:int=3):
        """Plot the persistence diagram.
        
        Parameters
        ----------
        persistence_pairs : list
            A list of (birth, death, dimension) tuples
        max_dimension : int
            Maximum dimension to include in the plot
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        """
        # Separate points by dimension
        points_by_dim = {p: [] for p in range(max_dimension + 1)}
        
        for birth, death, dim in persistence_pairs:
            if 0 <= dim <= max_dimension:  # Ensure dimension is within range
                points_by_dim[dim].append((birth, death))
        
        # Plot
        fig = plt.figure()
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        markers = ['o', 's', '^', 'D', 'x']
        
        # Find max value for setting plot limits
        max_val = 0
        for dim in points_by_dim:
            for birth, death in points_by_dim[dim]:
                if death != float('inf'):
                    max_val = max(max_val, death)
                max_val = max(max_val, birth)
        
        max_val *= 1.1  # Add some margin
        
        # Plot points for each dimension
        for dim in points_by_dim:
            births = []
            deaths = []
            
            for birth, death in points_by_dim[dim]:
                births.append(birth)
                # Handle infinite persistence by plotting at the top
                if death == float('inf'):
                    deaths.append(max_val)
                else:
                    deaths.append(death)
            
            if births:  # Only plot if there are points
                plt.scatter(births, deaths, color=colors[dim % len(colors)], 
                        marker=markers[dim % len(markers)], 
                        label=f'H_{dim}', alpha=0.7)
        
        # Plot the diagonal
        plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
        plt.xlim(left=-0.25)
        
        plt.xlabel('Birth')
        plt.ylabel('Death')
        plt.title('Persistence Diagram')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        return fig