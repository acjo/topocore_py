"""Filtration code."""

from pathlib import Path
from types import NoneType
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm import tqdm

from topocore.topocore import SimplicialComplex


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
        sub_sample: Optional[int] = 20,
    ) -> None:
        self.max_dimension = max_dimension
        self.file_id = file_id
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

        filtration_complexes: list[SimplicialComplex] = []

        idx = len(filtration_values) if sub_sample is None else sub_sample

        with tqdm(
            total=len(filtration_values[:idx]),
            desc=f"Building filtrations...",
        ) as loop:
            for threshold in filtration_values[:idx]:
                complex = SimplicialComplex.from_vietoris_rips_complex(
                    threshold, distance_matrix, max_dimension=max_dimension
                )

                filtration_complexes.append(complex)

                loop.update()

        self.filtration_complexes: list[SimplicialComplex] = (
            filtration_complexes
        )

        return
