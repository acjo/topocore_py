from matplotlib import pyplot as plt

from topocore import VRFiltration

if __name__ == "__main__":
    filtration = VRFiltration(max_dimension=3, sub_sample=500)
    persistence_pairs = filtration.compute_persistent_homology()
    filtration.plot_persistence_diagram(persistence_pairs)
    plt.show()
