import sys
sys.path.append('./src/')
import numpy as np
import matplotlib.pyplot as plt
from beta_rate import *

def labeling(my_list):
    """
    Turn list into new numerical list.
    """
    new_list = []
    unique = list(np.unique(my_list))
    for x in my_list:
        new_list.append(unique.index(x))
    return new_list

def data_statistics(data, species, is_H, resnames, in_backbone, what_mask='test_mask'):
    """
    Print out train/test dataset statistics.
    """
    print("--- Statistics for:",what_mask)
    species_counts = {'C':0,'O':0,'N':0,'H':0,'S':0,'heavy':0}
    atoms_per_resname = {'ALA':0,'ARG':0,'CYS':0,'GLU':0,'GLN':0,'THR':0,'TYR':0,'VAL':0,'ILE':0,'LEU':0,'PRO':0,'MET':0,'ASN':0,'ASP':0,'LYS':0,'PHE':0,'GLY':0,'SER':0,'TRP':0,'HIS':0}
    mask = data[what_mask]
    indeces=[]
    for j,tf in enumerate(mask):
        if tf == True: 
            atoms_per_resname[resnames[j]] += 1
    print(f"{atoms_per_resname=}")   

def plot_dict(dict0,dict_std,features,max_feat=None,png_name='Rates.png'):
    """                                                                                                                                                                            
    Plot errorbars
    """
    if max_feat == None: max_feat = len(dict0)
    fig = plt.figure(1, figsize=(4, 4))
    plt.rcParams.update({'font.size': 6.8})
    plt.errorbar(np.linspace(0,max_feat,max_feat),dict0[:max_feat],yerr=dict_std,fmt='none',elinewidth=1,ecolor='C0')
    plt.scatter(np.linspace(0,max_feat,max_feat),dict0[:max_feat],color='C0')
    plt.ylim(0.,1.)
    plt.ylabel('Rate')
    plt.xticks(np.linspace(0,max_feat,max_feat),np.array(features)[:max_feat],rotation=90)
    fig.savefig(png_name,dpi=150)
    plt.clf()
    print("DONE:",png_name)
    print()

def results(mask, pred, indeces, labels, species, is_H, resnames, in_backbone, inpname):
    """
    Plot prediction accuracy for all nodes, and write PDB with temperature factor according to prediction success label.
    """
    print("================ Results ================")
    successes = np.ones(len(mask))
    for i,pr in enumerate(list(pred)):
        success_check = int(pr)==labels[indeces[i]]
        successes[indeces[i]] = success_check*2 # *2 so that 0 will be wrongly classified, 1 not part of test set, and 2 correctly classified
        if i<5:
            print(f"{pr=} {indeces[i]=} {labels[indeces[i]]=} {species[indeces[i]]=} {is_H[indeces[i]]=} {resnames[indeces[i]]=} {success_check=}") # {list(labels).index[indeces[i]]=}")
    set_beta(successes,inpname)
    print()

def results_per_sel(pred, 
                    indeces, labels,
                    species, is_H, resnames, in_backbone, 
                    pdbid, output_directory, 
                    what_sel='resnames'):
    """
    Print and plot accuracies grouped by selection
    """
    print("================ Results per",what_sel,"================")
    if what_sel=='species': all_sel=['C','O','N','H','S']
    if what_sel=='resnames': all_sel=['ALA','ARG','CYS','GLU','GLN','THR','TYR','VAL','ILE','LEU','PRO','MET','ASN','ASP','LYS','PHE','GLY','SER','TRP','HIS']
    if what_sel=='in_backbone': all_sel=['backbone','side-chain']
    counts_per_sel = {}
    succ_per_sel = {}
    succ2_per_sel = {}
    std_per_sel = {}
    for sel in all_sel:
        counts_per_sel[sel] = 0
        succ_per_sel[sel] = 0.
        succ2_per_sel[sel] = 0.
        std_per_sel[sel] = 0.
    for i,pr in enumerate(list(pred)):
        success_check= int(pr)==labels[indeces[i]]
        if what_sel=='species': sel = species[indeces[i]]
        if what_sel=='resnames': sel = resnames[indeces[i]]
        if what_sel=='in_backbone': sel = in_backbone[indeces[i]]
        counts_per_sel[sel] += 1
        succ_per_sel[sel] += success_check
        succ2_per_sel[sel] += (success_check)**2
    for sel in all_sel:
        succ_per_sel[sel] /= counts_per_sel[sel]
        succ2_per_sel[sel] /= counts_per_sel[sel]
        std_per_sel[sel] = np.sqrt(succ2_per_sel[sel]-succ_per_sel[sel]*succ_per_sel[sel])
        print("Rate for {}: {} +- {}".format(sel,succ_per_sel[sel],std_per_sel[sel])) 
    plot_dict(list(succ_per_sel.values()),list(std_per_sel.values()),features=list(succ_per_sel.keys()),max_feat=None,png_name=output_directory+'Rates_per_'+what_sel+'_'+pdbid+'.png')
