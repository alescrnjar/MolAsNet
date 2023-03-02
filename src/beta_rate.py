import numpy as np
import MDAnalysis as mda

def set_beta(successes,inpname='6eqe_protein.mol2'):
    u=mda.Universe(inpname.replace('mol2','pdb'))
    prot=u.select_atoms('protein')
    betas=[]
    #for i_at,at in prot.atoms.ids:
    # https://www.mdanalysis.org/MDAnalysisTutorial/writing.html
    u.atoms.tempfactors = np.array(successes)
    outname=inpname.replace('.mol2','')+'_beta.pdb'
    #with MDAnalysis.Writer(outname, multiframe=True, bonds=None, n_atoms=u.atoms.n_atoms) as PDB:
    with mda.Writer(outname) as PDB:
        PDB.write(u.atoms)
    print(outname+" written.")

if __name__ == "__main__":   
    import random
    random_labels=[]
    for i in range(4199):
        random_labels.append(random.randint(0,1))
    set_beta(random_labels)

