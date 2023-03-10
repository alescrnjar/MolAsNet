import networkx as nx

def protein_graph(inpname='6eqe_protein.mol2'):
    G = nx.Graph() 
    inpf=open(inpname,'r')

    in_atoms = False
    in_bonds = False
    species_counts = {'C':0,'O':0,'N':0,'H':0,'S':0,'heavy':0}
    atoms_per_resname = {'ALA':0,'ARG':0,'CYS':0,'GLU':0,'GLN':0,'THR':0,'TYR':0,'VAL':0,'ILE':0,'LEU':0,'PRO':0,'MET':0,'ASN':0,'ASP':0,'LYS':0,'PHE':0,'GLY':0,'SER':0,'TRP':0,'HIS':0}
    for il,line in enumerate(inpf.readlines()):
        if il==2:
            number_of_atoms = int(line.split()[0])
            number_of_bonds = int(line.split()[1])
        if ('@<TRIPOS>BOND' in line): 
            in_atoms=False
        if ('@<TRIPOS>SUBSTRUCTURE' in line): 
            in_bonds = False
        if in_atoms == True:
            if (len(line.split())>5):
                at_id = int(line.split()[0])
                atom_name = line.split()[1]
                spec = atom_name[0]
                resn = line.split()[7]
                resn1 = resn
                if (resn=='HIE' or resn=='HID' or resn=='HIP'): resn1 = 'HIS'
                if (resn=='CYX'): resn1 = 'CYS'
                if (spec != 'H'):
                    is_H = 'heavy'
                else:
                    is_H = 'H'
                if (atom_name=='CA' or atom_name=='C' or atom_name=='N' or atom_name=='O'):
                    in_backbone = 'backbone'
                else:
                    in_backbone = 'side-chain'
                species_counts[spec] += 1
                atoms_per_resname[resn1] += 1
                G.add_node(at_id, species=spec, is_H=is_H, resname=resn, in_backbone=in_backbone)
        if in_bonds==True:
            if len(line)>1:
                n1 = int(line.split()[1])
                n2 = int(line.split()[2])
                if (n1 in G.nodes and n2 in G.nodes):
                    G.add_edge(int(n1), int(n2))
        if ('@<TRIPOS>ATOM' in line):
            in_atoms = True
        if ('@<TRIPOS>BOND' in line):
            in_bonds = True
    inpf.close()
    print("Number of nodes:",G.number_of_nodes(),"(atoms in mol2 file:",number_of_atoms,")")
    print("Number of edges:",G.number_of_edges(),"(bonds in mol2 file:",number_of_bonds,")")
    print(f"{species_counts=}")
    print(f"{atoms_per_resname=}")
    return G

