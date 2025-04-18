"""Persistent Homology Implementation."""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from topocore.filtration import VRFiltration


def build_persistence_boundary_matrix(filtration:"VRFiltration") ->tuple[np.ndarray, list]:
    """Build the boundary matrix for the persistence algorithm.
    
    Parameters
    ----------
    filtration : VRFiltration
        Vietoris-Rips filtration object
        
    Returns
    -------
    D : np.ndarray
        The boundary matrix for the persistence algorithm
    simplex_info : list
        A list of (filtration_index, dimension, simplex) for each column
    """
    # Ensure all simplicial complexes have their simplices in list form
    for complex in filtration.filtration_complexes:
        if not hasattr(complex, 'simplices_list'):
            complex.set_simplices_as_lists()
    
    # Track when each simplex first appears in the filtration
    birth_times = {}  # Maps simplex to its birth filtration index
    dimensions = {}   # Maps simplex to its dimension
    
    for filt_idx, complex in enumerate(filtration.filtration_complexes):
        for p in range(complex.k + 1):
            for simplex in complex.simplices_list[p]:
                # Create a hashable representation of the simplex
                simplex_key = tuple(sorted(simplex))
                
                # Record the birth time if we haven't seen this simplex before
                if simplex_key not in birth_times:
                    birth_times[simplex_key] = filt_idx
                    dimensions[simplex_key] = p
    
    # Create a list of all unique simplices with their birth times and dimensions
    all_simplices = [(birth_times[s], dimensions[s], s) for s in birth_times]
    
    # Sort by the criteria:
    # 1. Sequence (filtration index)
    # 2. Ascending order of p (dimension)
    # 3. For fixed p, sorted by key
    all_simplices.sort(key=lambda x: (x[0], x[1], x[2]))
    
    # Create a mapping from simplex to column index
    simplex_to_col = {s: i for i, (_, _, s) in enumerate(all_simplices)}
    
    # Initialize the boundary matrix
    n = len(all_simplices)
    D = np.zeros((n, n), dtype=int)
    
    # Fill in the boundary matrix
    for col, (_, p, simplex) in enumerate(all_simplices):
        if p == 0:  # 0-simplices have no boundary
            continue
        
        # Compute the (p-1)-faces of the simplex
        for i in range(len(simplex)):
            face = simplex[:i] + simplex[i+1:]
            face_key = tuple(sorted(face))
            
            if face_key in simplex_to_col:
                row = simplex_to_col[face_key]
                D[row, col] = 1
    
    return D, all_simplices

def reduce_boundary_matrix(D:np.ndarray) -> tuple[np.ndarray, dict]:
    """Reduce the boundary matrix using the standard persistence algorithm.
    
    Parameters
    ----------
    D : np.ndarray
        The boundary matrix for the persistence algorithm
    
    Returns
    -------
    R : np.ndarray
        The reduced boundary matrix
    low : dict
        Maps column index to row index of the lowest 1
    """
    R = D.copy()
    low = {}  # Maps column index to row index of the lowest 1
    
    # Initialize the lowest ones in each column
    for j in range(R.shape[1]):
        nonzero_rows = np.where(R[:, j] == 1)[0]
        if len(nonzero_rows) > 0:
            low[j] = int(nonzero_rows[-1])  # Get the highest index (lowest 1)
    
    # Reduce the matrix
    for j in range(R.shape[1]):
        while j in low and any(k for k in range(j) if k in low and low[k] == low[j]):
            # Find the leftmost column with the same lowest 1
            k = min(i for i in range(j) if i in low and low[i] == low[j])
            
            # Add column k to column j (Z/2Z addition)
            R[:, j] = (R[:, j] + R[:, k]) % 2
            
            # Update the lowest 1 in column j
            nonzero_rows = np.where(R[:, j] == 1)[0]
            if len(nonzero_rows) > 0:
                low[j] = int(nonzero_rows[-1])
            else:
                low.pop(j, None)
    
    return R, low


def extract_persistence_pairs(R, low, simplex_info, filtration):
    """Extract persistence pairs from the reduced boundary matrix.
    
    Parameters
    ----------
    R : np.ndarray
        The reduced boundary matrix
    low : dict
        Maps column index to row index of the lowest 1
    simplex_info : list
        A list of (filtration_index, dimension, simplex) for each column
    filtration : VRFiltration
        Vietoris-Rips filtration object
    
    Returns
    -------
    persistence_pairs : list
        A list of (birth, death, dimension) tuples
    """
    persistence_pairs = []
    
    # Keep track of which rows are paired
    paired_rows = set(low.values())
    
    # Process birth-death pairs
    for j in low:
        i = low[j]
        
        birth_idx, dim_i, _ = simplex_info[i]
        death_idx, _, _ = simplex_info[j]
        
        birth_value = filtration.filtration_complexes[birth_idx].filtration_value
        death_value = filtration.filtration_complexes[death_idx].filtration_value
        
        # The homology dimension is dim_i (not dim_i - 1)
        homology_dim = dim_i
        
        persistence_pairs.append((birth_value, death_value, homology_dim))
    
    # Process classes that never die
    unpaired_cols = [j for j in range(R.shape[1]) 
                    if j not in low and all(R[i, j] == 0 for i in range(R.shape[0]))]
    
    for j in unpaired_cols:
        birth_idx, dim_j, _ = simplex_info[j]
        birth_value = filtration.filtration_complexes[birth_idx].filtration_value
        
        # The homology dimension is dim_j (not dim_j - 1)
        homology_dim = dim_j
        
        persistence_pairs.append((birth_value, float('inf'), homology_dim))
    
    return persistence_pairs