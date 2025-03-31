from topocore import SimplicialComplex


def SimplicialComplexA() -> SimplicialComplex:
    """Instantiate and return simplical complex for first example.

    Parameters
    ----------
    None

    Returns
    -------
    complex : SimplicialComplex
        returns the simplicial complex for the first example
    """
    # Create and populate complex A
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
    edges = [
        {"Cow", "Rabbit"},
        {"Cow", "Horse"},
        {"Cow", "Dog"},
        {"Rabbit", "Horse"},
        {"Rabbit", "Dog"},
        {"Horse", "Dog"},
        {"Fish", "Dolphin"},
        {"Fish", "Oyster"},
        {"Dolphin", "Oyster"},
        {"Broccoli", "Fern"},
        {"Broccoli", "Onion"},
        {"Broccoli", "Apple"},
        {"Fern", "Onion"},
        {"Fern", "Apple"},
        {"Onion", "Apple"},
    ]

    # Add 2-simplices (triangles)
    triangles = [
        {"Cow", "Rabbit", "Horse"},
        {"Cow", "Rabbit", "Dog"},
        {"Cow", "Horse", "Dog"},
        {"Rabbit", "Horse", "Dog"},
        {"Fish", "Dolphin", "Oyster"},
        {"Broccoli", "Fern", "Onion"},
        {"Broccoli", "Fern", "Apple"},
        {"Broccoli", "Onion", "Apple"},
        {"Fern", "Onion", "Apple"},
    ]

    for v in vertex_set:
        complex.add_simplex(v)

    for edge in edges:
        complex.add_simplex(edge)

    for triangle in triangles:
        complex.add_simplex(triangle)

    return complex


if __name__ == "__main__":
    complex = SimplicialComplexA()
    complex.set_simplices_as_lists()
    complex.verify_boundary_matrices()

    p_chain = [frozenset({"Apple"})]

    print(complex.compute_boundary_matrix(0))
    print(complex.compute_boundary(0, p_chain))
    print(complex.find_kernel_basis(0))
    print(complex.find_homologies(0))
    print(complex.compute_homology_ranks())
