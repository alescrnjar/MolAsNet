# https://towardsdatascience.com/a-beginners-guide-to-graph-neural-networks-using-pytorch-geometric-part-1-d98dc93e7742
import sys
sys.path.append('./src/')

import networkx as nx
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

import argparse 
from tensorboardX import SummaryWriter 
from functions import *
from protein_graph_data import * 
from plots_for_gnn import * 

import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split
import torch_geometric.transforms as T

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

parser = argparse.ArgumentParser()
# Input parameters
parser.add_argument('--dset', default='MolAsNet', type=str) 
parser.add_argument('--input_directory', default='./example_input/', type=str)
parser.add_argument('--pdbid', default='6eqe', type=str) # PDB ID of input protein

# Model parameters
parser.add_argument('--feats', default='degree', type=str) # Features for embeddings.
parser.add_argument('--classification', default='is_H', type=str) # Which labels set to use for node classification.
parser.add_argument('--n_epochs', default=5000, type=int) # Number of epochs for training
parser.add_argument('--n_hidden', default=256, type=int) # Hidden layers dimension
parser.add_argument('--learning_rate', default=1e-3, type=float) 
parser.add_argument('--test_size', default=0.20, type=float) # Train/test size ratio
parser.add_argument('--random_seed', default=42, type=int) 

# Output parameters
parser.add_argument('--log_freq', default=100, type=int) # Frequency for output
parser.add_argument('--output_directory', default='./example_output/', type=str) 

args = parser.parse_args()
print(f"{args=}")

if args.dset=='MolAsNet':
    G=protein_graph(args.input_directory+args.pdbid+'_protein.mol2')

    species = []
    is_H = []
    resnames = []
    in_backbone = []
    for i,node in enumerate(G.nodes):
        is_H.append(G.nodes[node]['is_H'])
        species.append(G.nodes[node]['species'])
        resnames.append(G.nodes[node]['resname'])
        in_backbone.append(G.nodes[node]['in_backbone'])
    
    if args.classification=='is_H': labels = labeling(is_H)
    if args.classification=='species': labels = labeling(species)
    if args.classification=='in_backbone': labels = labeling(in_backbone)
    labels=np.asarray(labels).astype(np.int64)
    
    # Make COO-format edges
    adj = nx.to_scipy_sparse_matrix(G).tocoo() 
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)

    # Select features for embeddings.
    if args.feats=='degree':
        embeddings = np.array(list(dict(G.degree()).values()))
    # Standardize features by removing the mean and scaling to unit variance. 
    scale = StandardScaler() 
    embeddings = scale.fit_transform(embeddings.reshape(-1,1))


#### Custom dataset
class MolDataset(InMemoryDataset):
    def __init__(self, transform=None):
        super(MolDataset, self).__init__('.', transform, None, None)
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

    
if args.dset=='MolAsNet':
    dataset = MolDataset()
data = dataset[0]
data_statistics(data, species, is_H, resnames, in_backbone, what_mask='train_mask')
data_statistics(data, species, is_H, resnames, in_backbone, what_mask='test_mask')

# GCN model
class GCN_Net(torch.nn.Module):
    def __init__(self):
        super(GCN_Net, self).__init__()
        self.conv1 = GCNConv(data.num_features, args.n_hidden)
        self.conv2 = GCNConv(args.n_hidden, args.n_hidden)
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

model = GCN_Net().to(device)
print("AC: model defined")

##### Train

torch.manual_seed(args.random_seed)

optimizer_name = "Adam"
optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=args.learning_rate)

def train():
    model.train()
    #loss = F.nll_loss(model()[data.train_mask], data.y[data.train_mask]) #negative log likelihood https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html
    loss = F.cross_entropy(model()[data.train_mask], data.y[data.train_mask]) # https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py (Link found in here: https://towardsdatascience.com/a-beginners-guide-to-graph-neural-networks-using-pytorch-geometric-part-1-d98dc93e7742)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach().item()    
   
@torch.no_grad()
def testing_stage():
    model.eval()
    logits = model()

    mask1 = data['train_mask']
    pred1 = logits[mask1].max(1)[1]
    acc1 = pred1.eq(data.y[mask1]).sum().item() / mask1.sum().item()

    mask = data['test_mask']
    indeces=[]
    for i,tf in enumerate(mask):
        if tf==True: indeces.append(i)
    pred = logits[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

    results(mask, pred, indeces, labels, species, is_H, resnames, in_backbone, inpname=args.input_directory+args.pdbid+'_protein.mol2')
    if args.classification=='is_H':
        for what_sel in ['species','resnames','in_backbone']:
            results_per_sel(pred, 
                            indeces, labels,
                            species, is_H, resnames, in_backbone, 
                            args.pdbid, 
                            args.output_directory, 
                            what_sel='species')
    return acc1,acc

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
train_acc,test_acc = testing_stage()
#confusion_matrix()
print("Test end.")


print('#' * 70)
print('Train Accuracy: %s' %train_acc )
print('Test Accuracy: %s' % test_acc)
print('#' * 70)

print("SCRIPT END.")
