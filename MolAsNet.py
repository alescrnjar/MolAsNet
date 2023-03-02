# https://towardsdatascience.com/a-beginners-guide-to-graph-neural-networks-using-pytorch-geometric-part-1-d98dc93e7742
import networkx as nx
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

import argparse #AC
from tensorboardX import SummaryWriter #AC
from protein_graph_data import * #AC
from plots_for_gnn import * #AC
from beta_rate import *

parser = argparse.ArgumentParser()
# Input parameters
parser.add_argument('--dset', default='HnonH', type=str) #karate 6eqe HnonH
parser.add_argument('--input_directory', default='./', type=str)
parser.add_argument('--pdbid', default='6eqe', type=str)

# Model parameters
parser.add_argument('--feats', default='degree', type=str) 
parser.add_argument('--classification', default='is_H', type=str) 
parser.add_argument('--n_epochs', default=5000, type=int) #orig: 200
parser.add_argument('--n_hidden', default=256, type=int) #128 #64 #orig: 16
parser.add_argument('--learning_rate', default=1e-3, type=float) #1e-2 #orig: 1e-1
parser.add_argument('--test_size', default=0.20, type=float) #orig: 0.30
parser.add_argument('--random_seed', default=42, type=int) #orig: 42

# Output parameters
parser.add_argument('--log_freq', default=100, type=int)
parser.add_argument('--output_directory', default='./', type=str)

#parser.add_argument('--', default=, type=) #orig: 

args = parser.parse_args()
print(f"{args=}")

def labeling(my_list):
    new_list=[]
    unique=list(np.unique(my_list))
    for x in my_list:
        new_list.append(unique.index(x))
    return new_list

if args.dset=='karate':
    # load graph from networkx library
    G = nx.karate_club_graph()
    #print(f"{G=}")
    #for i in G.nodes:
    #    print(G.nodes[i])
    #for i in G.edges:
    #    print(i)
    #    #print(G.edges[i])

    # retrieve the labels for each node
    labels = np.asarray([G.nodes[i]['club'] != 'Mr. Hi' for i in G.nodes]).astype(np.int64)

    # create edge index from
    #AC convert into COO format
    adj = nx.to_scipy_sparse_matrix(G).tocoo() 
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    #print(f"{edge_index=}")

    # using degree as embedding # Embeddings or numerical representations for the nodes
    # The degree of a node is the number of connections that it has to other nodes in the network. 
    embeddings = np.array(list(dict(G.degree()).values()))
    print(f"{G.degree=}")
    print(f"{embeddings=}")
    print(f"{embeddings.shape=}")

    # normalizing degree values
    scale = StandardScaler() # Standardize features by removing the mean and scaling to unit variance. # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    embeddings = scale.fit_transform(embeddings.reshape(-1,1))
    print(f"{embeddings=}")
    print(f"{embeddings.shape=}")
elif args.dset=='HnonH':
    G=protein_graph(args.pdbid+'_protein.mol2')
    #G=protein_graph('6dg8_protein.mol2')
    species=[]
    is_H=[]
    resnames=[]
    in_backbone=[]
    for i,node in enumerate(G.nodes):
        #if G.nodes[node]['species']=='C': species.append(0)
        #if G.nodes[node]['species']=='O': species.append(1)
        #if G.nodes[node]['species']=='N': species.append(2)
        #if G.nodes[node]['species']=='H': species.append(3)
        #if G.nodes[node]['species']=='S': species.append(4)
        #if G.nodes[node]['is_H']=='H': is_H.append(0)
        #if G.nodes[node]['is_H']=='heavy': is_H.append(1)
        is_H.append(G.nodes[node]['is_H'])
        species.append(G.nodes[node]['species'])
        resnames.append(G.nodes[node]['resname'])
        in_backbone.append(G.nodes[node]['in_backbone'])
    if args.classification=='is_H': labels=labeling(is_H)
    if args.classification=='species': labels=labeling(species)
    if args.classification=='in_backbone': labels=labeling(in_backbone)
    labels=np.asarray(labels).astype(np.int64)
    adj = nx.to_scipy_sparse_matrix(G).tocoo() 
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    if args.feats=='degree':
        embeddings = np.array(list(dict(G.degree()).values()))
    # normalizing degree values
    scale = StandardScaler() # Standardize features by removing the mean and scaling to unit variance. # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    embeddings = scale.fit_transform(embeddings.reshape(-1,1))
"""
elif args.dset=='6eqe':
    G=protein_graph(args.pdbid+'_protein.mol2')
    #for i,node in enumerate(G.nodes):                                                                                                                 
    #    print(G.nodes[node]['species'])
    species=[]
    is_H=[]
    resnames=[]
    in_backbone=[]
    for i,node in enumerate(G.nodes):
        #print(G.nodes[node])
        if G.nodes[node]['species']=='C': species.append(0)
        if G.nodes[node]['species']=='O': species.append(1)
        if G.nodes[node]['species']=='N': species.append(2)
        if G.nodes[node]['species']=='H': species.append(3)
        if G.nodes[node]['species']=='S': species.append(4)
        if G.nodes[node]['is_H']=='H': is_H.append(0)
        if G.nodes[node]['is_H']=='heavy': is_H.append(1)
        resnames.append(G.nodes[node]['resname'])
        in_backbone.append(G.nodes[node]['in_backbone'])
    labels=species
    labels=np.asarray(labels).astype(np.int64)
    #print(f"{labels=}")
    adj = nx.to_scipy_sparse_matrix(G).tocoo() 
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    #print(f"{edge_index=}")
    # using degree as embedding # Embeddings or numerical representations for the nodes
    # The degree of a node is the number of connections that it has to other nodes in the network.
    if args.feats=='degree':
        embeddings = np.array(list(dict(G.degree()).values()))
    # normalizing degree values
    scale = StandardScaler() # Standardize features by removing the mean and scaling to unit variance. # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    embeddings = scale.fit_transform(embeddings.reshape(-1,1))
    #print(f"{embeddings=}")
"""

#### Custom dataset

import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split
import torch_geometric.transforms as T


# custom dataset
class ProteinDataset(InMemoryDataset):
    def __init__(self, transform=None):
        super(ProteinDataset, self).__init__('.', transform, None, None)
        data = Data(edge_index=edge_index)
        data.num_nodes = G.number_of_nodes()
        # embedding 
        data.x = torch.from_numpy(embeddings).type(torch.float32) #x: node features
        # labels
        y = torch.from_numpy(labels).type(torch.long) #y: node labels
        data.y = y.clone().detach()
        data.num_classes = 5
        # splitting the data into train, validation and test
        X_train, X_test, y_train, y_test = train_test_split(pd.Series(list(G.nodes())), 
                                                            pd.Series(labels),
                                                            test_size=args.test_size, 
                                                            random_state=args.random_seed)
        n_nodes = G.number_of_nodes()
        # create train and test masks for data
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[X_train.index] = True
        test_mask[X_test.index] = True
        data['train_mask'] = train_mask
        data['test_mask'] = test_mask
        self.data, self.slices = self.collate([data])

    def _download(self):
        return
    
    def _process(self):
        return
    
    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class HydrogenDataset(InMemoryDataset):
    def __init__(self, transform=None):
        super(HydrogenDataset, self).__init__('.', transform, None, None)
        data = Data(edge_index=edge_index)
        data.num_nodes = G.number_of_nodes()
        # embedding 
        data.x = torch.from_numpy(embeddings).type(torch.float32) #x: node features
        # labels
        y = torch.from_numpy(labels).type(torch.long) #y: node labels
        data.y = y.clone().detach()
        data.num_classes = 2
        # splitting the data into train, validation and test
        X_train, X_test, y_train, y_test = train_test_split(pd.Series(list(G.nodes())), 
                                                            pd.Series(labels),
                                                            test_size=args.test_size, 
                                                            random_state=args.random_seed)
        n_nodes = G.number_of_nodes()
        # create train and test masks for data
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[X_train.index] = True
        test_mask[X_test.index] = True
        data['train_mask'] = train_mask
        data['test_mask'] = test_mask
        self.data, self.slices = self.collate([data])

    def _download(self):
        return
    
    def _process(self):
        return
    
    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class KarateDataset(InMemoryDataset):
    def __init__(self, transform=None):
        super(KarateDataset, self).__init__('.', transform, None, None)
        data = Data(edge_index=edge_index)
        data.num_nodes = G.number_of_nodes()
        # embedding 
        data.x = torch.from_numpy(embeddings).type(torch.float32) #x: node features
        #print(f"{embeddings=}")
        #print(f"{data.x=}")
        # labels
        y = torch.from_numpy(labels).type(torch.long) #y: node labels
        #print(f"{labels=}")
        #print(f"{data.y=}")
        data.y = y.clone().detach()
        data.num_classes = 2
        # splitting the data into train, validation and test
        X_train, X_test, y_train, y_test = train_test_split(pd.Series(list(G.nodes())), 
                                                            pd.Series(labels),
                                                            test_size=0.30, 
                                                            random_state=42)
        #print(f"{X_train=}")
        #print(f"{y_train=}")
        #print(f"{X_test=}")
        #print(f"{y_test=}")
        #print(f"{len(X_train)=}")
        #print(f"{len(y_train)=}")
        #print(f"{len(X_test)=}")
        #print(f"{len(y_test)=}")
        n_nodes = G.number_of_nodes()
        # create train and test masks for data
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[X_train.index] = True
        test_mask[X_test.index] = True
        data['train_mask'] = train_mask
        data['test_mask'] = test_mask
        self.data, self.slices = self.collate([data])

    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

def data_statistics(data,what_mask='test_mask'):
    print("--- Statistics for:",what_mask)
    species_counts={'C':0,'O':0,'N':0,'H':0,'S':0,'heavy':0}
    atoms_per_resname={'ALA':0,'ARG':0,'CYS':0,'GLU':0,'GLN':0,'THR':0,'TYR':0,'VAL':0,'ILE':0,'LEU':0,'PRO':0,'MET':0,'ASN':0,'ASP':0,'LYS':0,'PHE':0,'GLY':0,'SER':0,'TRP':0,'HIS':0}
    mask = data[what_mask]
    indeces=[]
    for j,tf in enumerate(mask):
        if tf==True: 
            #print(f"{species[j]=} {is_H[j]=} {resnames[j]=}")
            #species_counts[species[j]]+=1
            atoms_per_resname[resnames[j]]+=1
    #print(f"{species_counts=}")
    print(f"{atoms_per_resname=}")   

    
if args.dset=='karate':
    dataset = KarateDataset()
elif args.dset=='6eqe':
    dataset = ProteinDataset()
elif args.dset=='HnonH':
    dataset = HydrogenDataset()
data = dataset[0]
data_statistics(data,what_mask='train_mask')
data_statistics(data,what_mask='test_mask')
#print(f"{data.train_mask=}")
#print(f"{X_train=} {X_test=} {y_train=} {y_test=}")

#### Graph Convo Network

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# GCN model with 2 layers 
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.num_features, args.n_hidden)
        self.conv2 = GCNConv(args.n_hidden, int(data.num_classes))

    def forward(self):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# GCN model with 3 layers 
class Net3(torch.nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = GCNConv(data.num_features, args.n_hidden)
        self.conv2 = GCNConv(args.n_hidden, args.n_hidden)
        #self.conv3 = GCNConv(args.n_hidden, int(data.num_classes))
        self.conv3 = GCNConv(args.n_hidden, args.n_hidden)
        self.conv4 = GCNConv(args.n_hidden, int(data.num_classes))

    def forward(self):
        x, edge_index = data.x, data.edge_index
        #
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        #
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        #
        #x = self.conv3(x, edge_index)
        #return F.log_softmax(x, dim=1)
        #
        x = F.relu(self.conv3(x, edge_index))
        x = F.dropout(x, training=self.training)
        #
        x = self.conv4(x, edge_index)
        return F.log_softmax(x, dim=1)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device='cpu'
print(f"{device=}")
data =  data.to(device)
#print(f"AC: {data=}")

#model = Net().to(device)
model = Net3().to(device)
print("AC: model defined")
##### Train

torch.manual_seed(args.random_seed)

optimizer_name = "Adam"
optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=args.learning_rate)

"""
def train():
    model.train()
    optimizer.zero_grad()
    #print(f"AC: {data.train_mask=}")
    #print(f"AC: {data.x=} {data.y=} {len(data.x)=} {len(data.y)=}")
    #print(f"{model()[data.train_mask]=}")
    #print(f"{data.y[data.train_mask]=}")
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward() #negative log likelihood https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html
    optimizer.step()
"""
#""
def train():
    model.train()
    """
    if args.dset=='HnonH':
        #loss = torch.nn.BCELoss(model()[data.train_mask], data.y[data.train_mask]) #in realta anche Karate aveva solo 2 classi e usava NLL!!!
        loss = F.nll_loss(model()[data.train_mask], data.y[data.train_mask]) #negative log likelihood https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html
    else:
        loss = F.nll_loss(model()[data.train_mask], data.y[data.train_mask]) #negative log likelihood https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html
    """
    loss = F.cross_entropy(model()[data.train_mask], data.y[data.train_mask]) # https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py (Link found in here: https://towardsdatascience.com/a-beginners-guide-to-graph-neural-networks-using-pytorch-geometric-part-1-d98dc93e7742)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach().item()
#""
"""
def train(): # https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py (Link found in here: https://towardsdatascience.com/a-beginners-guide-to-graph-neural-networks-using-pytorch-geometric-part-1-d98dc93e7742)
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)
"""
"""
def train():
    loss = loss_fn(pred, y_expected)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss += loss.detach().item()
    return total_loss / len(dataloader)
"""
"""

def train():
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.y.to(device)
	loss = crit(output, label)
        loss.backward()
	loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)

device = torch.device('cuda')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
crit = torch.nn.BCELoss()
train_loader = DataLoader(train_dataset, batch_size=batch_size)
"""

@torch.no_grad()
def test():
    model.eval()
    logits = model()

    mask1 = data['train_mask']
    #print(f"{mask1=}")
    pred1 = logits[mask1].max(1)[1]
    #print(f"{pred1=}")
    acc1 = pred1.eq(data.y[mask1]).sum().item() / mask1.sum().item()
    
    mask = data['test_mask']
    #print(f"{mask=}")
    pred = logits[mask].max(1)[1]
    #print(f"{pred=}")
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    return acc1,acc

def plot_dict(dict0,dict_std,features,max_feat=None,png_name='Rates.png'):
    """                                                                                                                                                                            
    Plot errorbars
    """
    if max_feat==None: max_feat=len(dict0)
    fig = plt.figure(1, figsize=(4, 4))
    plt.rcParams.update({'font.size': 6.8})
    #plt.errorbar(np.linspace(0,max_feat,max_feat),dict0[:max_feat].cpu().numpy(),yerr=dict_std,fmt='none',elinewidth=1,ecolor='C0')
    #plt.scatter(np.linspace(0,max_feat,max_feat),dict0[:max_feat].cpu().numpy(),color='C0')
    plt.errorbar(np.linspace(0,max_feat,max_feat),dict0[:max_feat],yerr=dict_std,fmt='none',elinewidth=1,ecolor='C0')
    plt.scatter(np.linspace(0,max_feat,max_feat),dict0[:max_feat],color='C0')
    #plt.xlabel('Feature')                                                                                                                                                         
    plt.ylim(0.,1.)
    plt.ylabel('Rate')
    #plt.xticks(np.arange(max_feat),features[:max_feat],rotation=90)
    plt.xticks(np.linspace(0,max_feat,max_feat),np.array(features)[:max_feat],rotation=90)
    fig.savefig(png_name,dpi=150)
    plt.clf()
    print("DONE:",png_name)
    print()
    
def results(mask, pred, indeces):
    print("================ Results ================")
    successes=np.ones(len(mask))
    mysum=0.0
    count=0
    for i,pr in enumerate(list(pred)):
        count+=1
        success_check=int(pr)==labels[indeces[i]]
        successes[indeces[i]]=success_check*2 #*2 so that 0 will be wrongly classified, 1 not part of test set, and 2 correctly classified
        if success_check: mysum+=1
        if i<4:
            print(f"{pr=} {indeces[i]=} {labels[indeces[i]]=} {species[indeces[i]]=} {is_H[indeces[i]]=} {resnames[indeces[i]]=} {success_check=}") # {list(labels).index[indeces[i]]=}")
    set_beta(successes,inpname=args.pdbid+'_protein.mol2')
    mysum/=count
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    #print(f"{mysum=} {acc=}")
    print(f"{acc=}")
    print()

def results_per_sel(pred, indeces, what_sel='resnames'):
    print("================ Results per",what_sel,"================")
    if what_sel=='species': all_sel=['C','O','N','H','S']
    if what_sel=='resnames': all_sel=['ALA','ARG','CYS','GLU','GLN','THR','TYR','VAL','ILE','LEU','PRO','MET','ASN','ASP','LYS','PHE','GLY','SER','TRP','HIS']
    if what_sel=='in_backbone': all_sel=['backbone','side-chain']
    counts_per_sel={}
    succ_per_sel={}
    succ2_per_sel={}
    std_per_sel={}
    for sel in all_sel:
        counts_per_sel[sel]=0
        succ_per_sel[sel]=0.
        succ2_per_sel[sel]=0.
        std_per_sel[sel]=0.
    for i,pr in enumerate(list(pred)):
        success_check=int(pr)==labels[indeces[i]]
        if what_sel=='species': sel=species[indeces[i]]
        if what_sel=='resnames': sel=resnames[indeces[i]]
        if what_sel=='in_backbone': sel=in_backbone[indeces[i]]
        counts_per_sel[sel]+=1
        succ_per_sel[sel]+=success_check
        succ2_per_sel[sel]+=(success_check)**2
    for sel in all_sel:
        succ_per_sel[sel]/=counts_per_sel[sel]
        succ2_per_sel[sel]/=counts_per_sel[sel]
        std_per_sel[sel]=np.sqrt(succ2_per_sel[sel]-succ_per_sel[sel]*succ_per_sel[sel])
        print("Rate for {}: {} +- {}".format(sel,succ_per_sel[sel],std_per_sel[sel])) 
    plot_dict(list(succ_per_sel.values()),list(std_per_sel.values()),features=list(succ_per_sel.keys()),max_feat=None,png_name='Rates_per_'+what_sel+'_'+args.pdbid+'.png')
    
def my_test():
    model.eval()
    logits = model()
    mask = data['test_mask']
    indeces=[]
    for i,tf in enumerate(mask):
        if tf==True: indeces.append(i)
    print(f"{mask.shape=}")
    pred = logits[mask].max(1)[1]
    results(mask, pred, indeces)
    if args.classification=='is_H':
        results_per_sel(pred, indeces, what_sel='species')
        results_per_sel(pred, indeces, what_sel='resnames')
        results_per_sel(pred, indeces, what_sel='in_backbone')
    
def confusion_matrix():
    if args.dset=='6eqe':
        classes=['C','O','N','H','S']
    elif args.dset=='HnonH':
        classes=['H','heavy']
    # Initialize confusion matrix                                        
    confusion_matrix = []
    for c1 in classes:
        confusion_matrix.append([])
        for c2 in classes:
            confusion_matrix[-1].append(0)
    n_predictions_per_class = {}
    for c2 in classes:
        n_predictions_per_class[c2] = 0
        
    model.eval()
    logits = model()
    mask = data['test_mask']
    indeces=[]
    for i,tf in enumerate(mask):
        if tf==True: indeces.append(i)
    pred = logits[mask].max(1)[1]
    for i,pr in enumerate(list(pred)):
        success_check=int(pr)==labels[indeces[i]]
        #confusion_matrix[classes.index(predicted_class)][classes.index(expected_class)] += 1
        #n_predictions_per_class[expected_class] += 1
        confusion_matrix[int(pr)][labels[indeces[i]]] += 1
        n_predictions_per_class[classes[int(pr)]] += 1
            
    """
    count=0
    rate=0
    for test in test_dataset:
        example=test[2]
        for t in range(3): # For each example sequence, test three randomly-assigned 
            count += 1
            predicted_class, expected_class , descending_predictions = evaluate(net, example)
            print("Predicted: {} Expected: {} ({})".format(predicted_class, expected_class, descending_predictions))
            rate += (predicted_class == expected_class)
            confusion_matrix[classes.index(predicted_class)][classes.index(expected_class)] += 1
            n_predictions_per_class[expected_class] += 1
    """
    
    # Normalize elements of prediction matrix    
    for c1 in classes:
        for c2 in classes:
            if n_predictions_per_class[c2] != 0 : confusion_matrix[classes.index(c1)][classes.index(c2)] /= n_predictions_per_class[c2]
    plot_confusion_matrix(confusion_matrix, classes, args.output_directory)

    #print("Success prediction rate:",rate/count)

#if __name__ == "__main__":   
print("Training start.")
#tensorboard --logdir=./
summary_writer = SummaryWriter(args.output_directory)
for epoch in range(args.n_epochs):
    #print(f"{epoch=}")
    #train()
    loss=train()
    if epoch%args.log_freq==0: print(f"{epoch=} {loss=}")
    summary_writer.add_scalar('Loss',torch.FloatTensor([loss]),global_step=epoch)
print("Training end.")

print("Test start.")
train_acc,test_acc = test()
my_test()
#confusion_matrix()
print("Test end.")


print('#' * 70)
print('Train Accuracy: %s' %train_acc )
print('Test Accuracy: %s' % test_acc)
print('#' * 70)

print("SCRIPT END.")
