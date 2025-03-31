"""Simplicial Complex B from homework."""

from topocore import SimplicialComplex


def SimplicialComplexB() -> SimplicialComplex:
    """Instantiate and return simplical complex for second example.

    Parameters
    ----------
    None

    Returns
    -------
    complex : SimplicialComplex
        returns the simplicial complex for the second example
    """
    # Create and populate complex B
    complex = SimplicialComplex()

    vertices = [
        "Cow",
        "Rabbit",
        "Horse",
        "Dog",
        "Fish",
        "Dolphin",
        "Oyster",
        "Broccoli",
        "Fern",
        "Onion",
        "Apple",
    ]

    vertex_set = [{v} for v in vertices]

    # Add 1-simplices (edges)
    edges_b = [
        {"Cow", "Rabbit"},
        {"Cow", "Fish"},
        {"Cow", "Oyster"},
        {"Cow", "Broccoli"},
        {"Cow", "Onion"},
        {"Cow", "Apple"},
        {"Rabbit", "Fish"},
        {"Rabbit", "Oyster"},
        {"Rabbit", "Broccoli"},
        {"Rabbit", "Onion"},
        {"Rabbit", "Apple"},
        {"Fish", "Oyster"},
        {"Fish", "Broccoli"},
        {"Fish", "Onion"},
        {"Fish", "Apple"},
        {"Oyster", "Broccoli"},
        {"Oyster", "Onion"},
        {"Oyster", "Apple"},
        {"Broccoli", "Onion"},
        {"Broccoli", "Apple"},
        {"Onion", "Apple"},
        {"Horse", "Dog"},
        {"Horse", "Dolphin"},
        {"Horse", "Fern"},
        {"Dog", "Dolphin"},
        {"Dog", "Fern"},
        {"Dolphin", "Fern"},
    ]

    # Add 2-simplices (triangles)
    triangles_b = [
        {"Cow", "Broccoli", "Apple"},
        {"Cow", "Onion", "Apple"},
        {"Rabbit", "Broccoli", "Apple"},
        {"Rabbit", "Onion", "Apple"},
        {"Fish", "Broccoli", "Apple"},
        {"Fish", "Onion", "Apple"},
        {"Oyster", "Broccoli", "Apple"},
        {"Oyster", "Onion", "Apple"},
    ]

    for v in vertex_set:
        complex.add_simplex(v)
    for edge in edges_b:
        complex.add_simplex(edge)
    for triangle in triangles_b:
        complex.add_simplex(triangle)

    return complex


if __name__ == "__main__":
    complex = SimplicialComplexB()
    complex.set_simplices_as_lists()
    complex.verify_boundary_matrices()

    p_chain = [frozenset({"Apple"})]

    print(complex.compute_boundary_matrix(0))
    print(complex.compute_boundary(0, p_chain))
    print(complex.find_kernel_basis(0))
    print(complex.find_homologies(0))
    print(complex.compute_homology_ranks())
