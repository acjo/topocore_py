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
