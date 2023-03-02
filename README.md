# MolAsNet

<!--AtomGuesser ProteinNode ProteinGraph NodeInProtein ProNode CONSHGuess OrganicGuesser TopGraph Mol2Grapher Link Tree GraTein PepGraph(esiste) Mol2NX Mol2Net(Esistente) Mol2Nex(libero) Pro2Net(esistente) MolAsNet(libero)
-->

<!-- TO DO: 
- Altre idee: bond length regressor: edge regressor
- SPM regressor
- t-SNE per spiegare il layer, dal paper di GCNConv!!!
- resname classifier: ma servirebbe un grafo enorme, perche molti resnames avrebbero poche entries
- propagazione degli errori e piu corretta, per l'accuracy per resn!
-->

MolAsNet is a Graph Convolutional Neural (GCN) Network, that takes the structure of a given protein as a graph and predicts whether atoms are hydrogens or heavy atoms (C, N, O, S): thus performing a node classification task. The protein structure is loaded as a .mol2 file, which provides information on the atoms identity as well as the bond network.

The code is adapted from this tutorial: https://towardsdatascience.com/a-beginners-guide-to-graph-neural-networks-using-pytorch-geometric-part-1-d98dc93e7742

The chosen default embeddings/numerical representations for the nodes is node degree.

<!--The architecture includes 4 GCNConv layers (first described by Kipf et al.: https://arxiv.org/abs/1609.02907).

The used loss function is Cross Entropy.-->

The provided example .mol2 file regards the crystal structure of a polyethylene terephthalate degrading hydrolase (PDB ID: 6EQE, https://www.rcsb.org/structure/6eqe). The .pdb file was downloaded already included hydrogens, and the software VMD was used to make a .mol2 file for the selection of protein atoms.

Throught the usage of the library MDAnalysis, an output .pdb is produced, whose atoms temperature factors beta take three values: 0 (red with beta coloring method in VMD) for incorrectly classified atom, 1 (white) for non classified atom (i.e. not part of the test set), 2 (blue) for correctly classified atom.

# Required Libraries

* numpy >= 1.21.5

* pandas >= 1.5.1

* torch_geometric >= 2.2.0

* torch >= 1.13.0

* networkx >= 2.8.4

* tensorboardX >= 2.11.2

* matplotlib >= 3.5.2

* MDAnalysis >= 2.2.0

<!--* sklearn: NON TROVATO CON PIP SHOW!
* tqdm >= 4.64.0 -->

# Case Study: 6EQE

<p align="center">
<img width="500" src=https://github.com/alescrnjar/MolAsNet/blob/main/example_output/Screenshot.png>
</p>


<!--
# Example Confusion Matrix
-->

