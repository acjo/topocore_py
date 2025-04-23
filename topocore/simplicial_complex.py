"""Abstract Simplical Complex Code."""

from collections import defaultdict
from itertools import combinations

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm

from topocore.linalg import image_mod2, nullspace_mod2, rref_mod2

rcParams["font.family"] = "serif"
rcParams["font.size"] = 15
rcParams["axes.labelsize"] = 10
rcParams["xtick.labelsize"] = 10
rcParams["ytick.labelsize"] = 10
rcParams["legend.fontsize"] = 10
rcParams["figure.figsize"] = (7, 7)
rcParams["figure.dpi"] = 200
rcParams["axes.titlesize"] = 15


class SimplicialComplex(object):
    """Simplical Complex Class.

    Prameters
    ---------
    None

    Attributes
    ----------
    simplices : dit[set | list]
        set of simplicies
    """

    def __init__(self) -> None:
        # create a dictionary that will map simplex dimension to the set of simplices
        self.simplices = defaultdict(lambda: set())
        # stores the maximum dimension of all p_simplices
        self.k = -1
        self.filtration_value: float = -1.0
        return

    @staticmethod
    def build_filtration_incrementally(
        distance_matrix: np.ndarray,
        thresholds: np.ndarray,
        max_dimension: int = 3,
    ):
        """Build a filtration of Vietoris-Rips complexes incrementally.

        Parameters
        ----------
        distance_matrix : np.array
            Pairwise distance matrix
        thresholds : np.array
            List of threshold values in ascending order
        max_dimension : int
            Maximum simplex dimension to include, default 3

        Returns
        -------
        filtration : list[SimplicialComplex]
            List of simplicial complexes forming a filtration
        """
        n = distance_matrix.shape[0]
        filtration = []

        # Keep track of which edges and higher simplices have been added
        # to avoid redundant checks in subsequent iterations
        added_edges = set()
        added_simplices = {k: set() for k in range(2, max_dimension + 1)}

        # Process each threshold value

        for threshold in thresholds:
            # Create a new complex or copy the previous one if it exists
            if not filtration:
                complex = SimplicialComplex()

                # Add all vertices (these are always included)
                for i in range(n):
                    complex.add_simplex({i})
            else:
                # Create a deep copy of the previous complex
                complex = SimplicialComplex()
                complex.k = filtration[-1].k

                # Copy existing simplices
                for dim, simplices in filtration[-1].simplices.items():
                    complex.simplices[dim] = simplices.copy()

            # Set filtration value
            complex.filtration_value = threshold

            # Add 1-simplices (edges) that satisfy the new threshold
            rows, cols = np.triu_indices(n, k=1)

            # Find edges that meet the threshold and haven't been added yet
            for i, j in zip(rows, cols):
                edge = frozenset({int(i), int(j)})
                if (
                    edge not in added_edges
                    and distance_matrix[i, j] <= threshold
                ):
                    complex.add_simplex(set(edge))
                    added_edges.add(edge)

            # Add higher-dimensional simplices
            if max_dimension >= 2:
                # For each potential simplex dimension
                for k in range(2, max_dimension + 1):
                    # We only need to check k-simplices whose (k-1)-faces are all in the complex
                    # Use the 1-skeleton (edges) to determine potential k-simplices

                    # Find all k+1 sized combinations of vertices that could form a k-simplex
                    # This is where we can be smarter than checking all combinations
                    potential_vertices = []

                    # Get all edges as pairs of vertices
                    edges = [
                        (tuple(e)[0], tuple(e)[1]) for e in complex.simplices[1]
                    ]

                    # Build graph from edges for faster checking
                    graph = defaultdict(set)
                    for v1, v2 in edges:
                        graph[int(v1)].add(int(v2))
                        graph[int(v2)].add(int(v1))

                    # Find k+1 cliques in the graph
                    # For large graphs, we'd use a more efficient algorithm
                    # but for simplicity, we'll use a basic approach

                    # Start with vertices
                    vertices = list(range(n))

                    # For each potential starting vertex
                    for start_vertex in vertices:
                        # Find all k-sized combinations from its neighbors
                        # that could form a (k+1)-clique with the start vertex
                        if len(graph[start_vertex]) >= k:
                            for neighbors in combinations(
                                graph[start_vertex], k
                            ):
                                # Check if these neighbors form a clique
                                if all(
                                    v2 in graph[v1]
                                    for v1, v2 in combinations(neighbors, 2)
                                ):
                                    # This is a (k+1)-clique: [start_vertex] + neighbors
                                    potential_vertices.append(
                                        (start_vertex,) + neighbors
                                    )

                    # Add new k-simplices
                    for vertices in potential_vertices:
                        simplex = frozenset(vertices)
                        if simplex not in added_simplices[k]:
                            # Check if all edges meet the threshold
                            row_idx, col_idx = np.asarray(
                                list(combinations(vertices, 2)), dtype=int
                            ).T

                            all_edges_valid = (
                                distance_matrix[row_idx, col_idx] <= threshold
                            ).all()

                            # If all edges are valid, add the simplex
                            if all_edges_valid:
                                complex.add_simplex(set(simplex))
                                added_simplices[k].add(simplex)

            # Set list representation for computing homology
            complex.set_simplices_as_lists()
            filtration.append(complex)

        return filtration

    def add_simplex(self, simplex: set) -> None:
        """Add simplex to simplices.

        Parameters
        ----------
        simplex : set
            a simplex to add to

        Returns
        -------
        None
        """
        # convert to frozen set for immutability and hashability.
        p = len(simplex) - 1
        self.simplices[p].add(frozenset(simplex))
        # update dimension of the complex if needed
        if p > self.k:
            self.k = p

        return

    def set_simplices_as_lists(self) -> None:
        """Convert simplex sets to simplex lists.

        This is done to preserve relative order
        """
        self._is_simplicial_complex()
        self.simplices_list = dict()
        for p, simp in self.simplices.items():
            simplex_list = [sorted(list(v)) for v in simp]
            self.simplices_list[p] = sorted(simplex_list)

        return

    def _get_faces(self, simplex: set, size: int) -> list[frozenset]:
        """Get faces of a given size.

        Parameters
        ----------
        simplex : set
            simplex to get the face of
        size : int
            what dimension the face should be.

        Returns
        -------
        faces : list of sets
            list of sets containing the faces.
        -------
        """
        return [frozenset(face) for face in combinations(simplex, size)]

    def compute_boundary_matrix(self, p: int) -> np.ndarray:
        """Compute the boundary matrix.

        Computes the boundary matrix from p to p-1
        if p == 0 (e.g. vertices) then this will return the corresponding
        ones matrix.

        Parameters
        ----------
        p : int
            the dimension to map from

        Returns
        -------
        delp : np.ndarray (M,N)
            The boundary matrix where M is the number of p-1 simplices and N is the number of p simplices


        Notes
        -----
        * If you modify the set, you will have to re-create the boundary matrix as the sets within p_simplices might change order (python sets are unordered)
        * Between python sessions, the boundaries matrices might not have the same form (as python sets are unordered), however, the boundaries will obviously be equivalent.
        """
        self._check_simplex_list()

        N = len(self.simplices_list[p])
        if p == 0:
            M = 1
            return np.zeros((M, N), dtype=int)
        else:
            M = len(self.simplices_list[p - 1])

        del_p = np.zeros((M, N), dtype=int)

        for col, simplex in enumerate(self.simplices_list[p]):
            for row, face in enumerate(self.simplices_list[p - 1]):
                # In Z/2Z homology, we need to check if face is EXACTLY a p-1 face of simplex
                if (
                    set(face).issubset(set(simplex))
                    and len(face) == len(simplex) - 1
                ):
                    del_p[row, col] = 1

        return del_p

    def compute_boundary(
        self, p: int, p_chain: list[list]
    ) -> tuple[np.ndarray, list[list]]:
        """Compute the boundary of a p-chain.

        Given a p-chain represented as a frozen set, compute the boundary.

        This is done in Mod2. e.g. 1+1 = 2

        Prameters
        ---------
        p : int
            The dimension of the simplicies in the p-chain
        p_chain : list of frozen sets
            the p_chain represented as a set of frozen sets.

        Returns
        -------
        coefficients : ndarray
            coefficients representing the coefficients in the formal sum
        basis : list of frozen sets
            the basis of the edges

        Raises
        ------
        KeyError
            if the p_chain is not inside the simplicial complex.
        ValueError
            if the elements in p_chain are not unique.
        AttributeError
            if simplices_list is not an attribute or if the length of simplices_list does not match the length of simplices.
        """
        self._check_simplex_list()
        p_simplices = self.simplices_list[p]
        p_chain = sorted([sorted(simplex) for simplex in p_chain])
        p_chain_set = set([frozenset(simplex) for simplex in p_chain])
        if p > 0:
            basis = self.simplices_list[p - 1]
        else:
            basis = [["emptyset"]]

        if len(p_chain_set) != len(p_chain):
            raise ValueError("Elements in p-chain are not unique")
        del p_chain_set
        if not all(simp in p_simplices for simp in p_simplices):
            raise KeyError(
                "the p-chain is not contained inside the simplicial complex."
            )

        # vector representation of the p_chain in the p-simplex basis
        vector_rep = np.zeros(len(p_simplices), dtype=int)
        for p_simplex in p_chain:
            index = p_simplices.index(p_simplex)
            vector_rep[index] = 1

        del_p = self.compute_boundary_matrix(p)

        boundary = del_p @ vector_rep

        if p == 0:
            boundary = np.array([0] * len(basis))
        else:
            boundary = np.mod(boundary, 2)

        return boundary, basis

    def find_kernel_basis(
        self, p: int
    ) -> tuple[list[list[frozenset]], list[np.ndarray]]:
        """Get a basis for the kernel of the boundary matrix ∂_p.

        Parameters
        ----------
        p : int
            Dimension of the boundary operator input

        Returns
        -------
        Z_p_basis : list list of  sets
            The basis of the kernel of the boundary operator
        Z_p_basis
            The basis vectors of the kernel

        Raises
        ------
        RuntimeError
            If the null space is incorrectly calculated.
        """
        self._check_simplex_list()

        if p == 0:
            basis = [self.simplices_list[0]]
            vecs = []
            for i in range(len(self.simplices_list[0])):
                v = np.zeros(len(self.simplices_list[0]), dtype=int)
                v[i] = 1

                vecs.append(v)
            return basis, vecs

        del_p = self.compute_boundary_matrix(p)

        basis_coefficients = nullspace_mod2(del_p)

        cycles = []
        for vec in basis_coefficients:
            if np.any(
                np.zeros(del_p.shape[0], dtype=int) != ((del_p @ vec) % 2)
            ):
                raise RuntimeError("Null space baiss incorrectly calcualted.")

            cycle = []
            for i, coef in enumerate(vec):
                if coef == 1:
                    simplex = self.simplices_list[p][i]
                    cycle.append(simplex)

            cycles.append(cycle)

        return cycles, basis_coefficients

    def find_homologies(self, p: int) -> list[list[frozenset]]:
        """Get The homoligies of dimension p.

        Parameters
        ----------
        p : int
            Dimension.

        Returns
        -------
        homolgies : list of  sets
            The homologies of dimension p

        """
        self._check_simplex_list()

        Z_p_cycles, Z_p_basis = self.find_kernel_basis(p)

        # if the dimension is the same as the dimension of the simplicial complex then im(∂_{p+1}) = 0 which means H_p = ker(∂_p)/im(∂_{p+1}) = ker(∂_p)
        if p == self.k:
            return Z_p_cycles

        # get boundary matrix.
        del_p_plus_1 = self.compute_boundary_matrix(p + 1)
        _, pivot_cols = rref_mod2(del_p_plus_1)
        rank_del_p_plus_1 = len(pivot_cols)

        Homologies = []
        for vec, cycle in zip(Z_p_basis, Z_p_cycles):
            # stack basis vector with boundary matrix
            augmented = np.column_stack((del_p_plus_1, vec))
            _, pivot_cols = rref_mod2(augmented)
            rank_augmented = len(pivot_cols)
            # if rank of the boundary operator [∂_{p+1}, b]
            # has increased, where b is a basis vector of the nullspace of ∂_{p}
            # that means b is NOT in the image of ∂_{p+1}, so it is in the homology
            if rank_augmented > rank_del_p_plus_1:
                Homologies.append(cycle)

        return Homologies

    def compute_homology_ranks(self):
        """Compute the ranks.

        of cycles, boundaries, and homology groups

        Parameters
        ----------
        boundary_matrices: List of boundary matrices [∂₀, ∂₁, ∂₂, ...]

        Returns
        -------
        ranks: Dict containing ranks of Z_n, B_n, and H_n for each dimension
        """
        max_dim = self.k
        ranks = {
            "Z": [0] * (max_dim + 1),
            "B": [0] * (max_dim + 1),
            "H": [0] * (max_dim + 1),
        }

        # Store boundary matrices with their shapes
        boundary_matrices = [
            self.compute_boundary_matrix(p) for p in range(self.k + 1)
        ]

        for p in range(max_dim + 1):
            # Get current boundary matrix
            del_p = boundary_matrices[p]

            # 1. Compute rank of cycles Z_n = ker(∂_n)
            if p == 0:
                # all C_0 chains map to zero, so the nullity of ∂_0 is just the
                # dimension of C_0.
                ranks["Z"][p] = del_p.shape[1]
            else:
                # For other dimensions, compute the nullity
                nullity = len(nullspace_mod2(boundary_matrices[p]))
                ranks["Z"][p] = nullity

            # 2. Compute rank of boundaries B_n = im(∂_{n+1})
            if p == self.k:
                ranks["B"][p] = 0
            else:
                del_n_plus_1 = boundary_matrices[p + 1]

                rank = len(image_mod2(del_n_plus_1))
                ranks["B"][p] = rank

                # Sanity check should never be needed if matrices are correct
                if ranks["B"][p] > ranks["Z"][p]:
                    print(
                        f"WARNING: B_{p} > Z_{p}! This is mathematically impossible."
                    )
                    print(f"∂{p+1} shape: {del_n_plus_1.shape}")
                    print(
                        f"B_{p} rank: {ranks['B'][p]}, Z_{p} rank: {ranks['Z'][p]}"
                    )

            # 3. Compute homology rank
            ranks["H"][p] = ranks["Z"][p] - ranks["B"][p]

        return ranks

    def visualize_complex(self) -> nx.Graph:
        """Display the complex as a NetworkX Graph.

        Returns
        -------
        G : NetworkX graph
            Graph containig vertices and edges
        """
        fig = plt.figure(facecolor="grey")
        ax = fig.add_subplot(111)
        G = nx.Graph()

        for s, p_simplicies in self.simplices.items():
            for simplex in p_simplicies:
                val = list(simplex)
                if s == 0:
                    G.add_node(val[0])
                elif s == 1:
                    G.add_edge(val[0], val[1])

        nx.draw_networkx(
            G,
            # pos=nx.layout.spring_layout(G, k=0.7),
            pos=nx.nx_agraph.graphviz_layout(G),
            node_size=150,
            font_size=8,
            with_labels=True,
            font_weight="bold",
            ax=ax,
        )
        return G

    def _check_simplex_list(self) -> None:
        if not hasattr(self, "simplices_list"):
            raise AttributeError(
                "You need to call set_simplices_as_lists before calling this function."
            )
        elif len(self.simplices_list) != len(self.simplices) or any(
            len(self.simplices[p]) != len(self.simplices_list[p])
            for p in self.simplices
        ):
            raise AttributeError(
                "You need to re-call set_simplices_as_lists before calling this function."
            )

        return

    def _is_simplicial_complex(self) -> None:
        """Check if the complex is actually a simplicial complex.

        The requriement of a simplicial complex is that the face of every p-simplex is also an element of the simplicial complex.

        For example for the triangle, [a,b,c] the following are required to be a part of the simplicial complex:
            [a,b]
            [a,c]
            [b,c]
            [a]
            [b]
            [c]

        Raises
        ------
        RuntimeError
            If the simplicial complex is not a complex.

        """
        keys = sorted(self.simplices.keys())

        for p_minus_1, p in zip(keys[:-1], keys[1:]):
            p_minus_1_simplcies = self.simplices[p_minus_1]
            for p_simplex in self.simplices[p]:
                for face in self._get_faces(p_simplex, p_minus_1 + 1):
                    if face not in p_minus_1_simplcies:
                        raise RuntimeError(
                            f"This is not a valid simplicial complex. The face {face} of the {p} simplex: {p_simplex} is not a member of the set of the {p_minus_1} simplices:\n{p_minus_1_simplcies}"
                        )

    def verify_boundary_matrices(self) -> None:
        """Verify that ∂ₙ₊₁ ∘ ∂ₙ = 0 for all n.

        Raises
        ------
        RuntimeError
            If boundary matrix composition fails
        """
        self._check_simplex_list()

        boundary_matrices = [
            self.compute_boundary_matrix(p) for p in range(self.k + 1)
        ]

        for p in range(self.k):
            del_p = boundary_matrices[p]
            del_n_plus_1 = boundary_matrices[p + 1]

            composition = del_p @ del_n_plus_1
            composition %= 2

            if np.any(composition != np.zeros_like(composition)):
                raise RuntimeError("Boundary Matrices Should Commute to zero.")

        return

    def verify_euler_characteristic(self) -> bool:
        """
        Verify that the sum of Betti numbers matches the Euler characteristic.

        The Euler characteristic can be computed in two ways:
        1. χ = Σ (-1)^i * number of i-simplices
        2. χ = Σ (-1)^i * β_i (where β_i is the ith Betti number)

        These two calculations should yield the same result.

        Returns
        -------
        bool
            True if the check passes, False otherwise
        """
        self._check_simplex_list()

        # Calculate Euler characteristic from simplex counts
        euler_from_simplices = 0
        for dim, simplices in self.simplices.items():
            euler_from_simplices += (-1) ** dim * len(simplices)

        # Calculate homology ranks
        ranks = self.compute_homology_ranks()
        betti_numbers = ranks["H"]

        # Calculate Euler characteristic from Betti numbers
        euler_from_betti = 0
        for dim, betti in enumerate(betti_numbers):
            euler_from_betti += (-1) ** dim * betti

        # Check if they match
        return euler_from_simplices == euler_from_betti
