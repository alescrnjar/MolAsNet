# https://towardsdatascience.com/a-beginners-guide-to-graph-neural-networks-using-pytorch-geometric-part-1-d98dc93e7742
import sys
sys.path.append('./src/')

import networkx as nx
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

import argparse 
from tensorboardX import SummaryWriter 
from gcn_models import *
#from functions import *
from protein_graph_data import * 
from plots_for_gnn import * 

parser = argparse.ArgumentParser()
# Input parameters
parser.add_argument('--what_graph', default='MolAsNet', type=str) 
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

#if __name__ == "__main__":   

args = parser.parse_args()
print(f"{args=}")

# Graph definition
if args.what_graph=='MolAsNet':
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
    
if args.what_graph=='MolAsNet':
    dataset = MolDataset(G, edge_index, embeddings, labels, args.test_size, args.random_seed)
data = dataset[0]
data_statistics(data, species, is_H, resnames, in_backbone, what_mask='train_mask')
data_statistics(data, species, is_H, resnames, in_backbone, what_mask='test_mask')

# Device setting
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device='cpu'
print(f"{device=}")
data =  data.to(device)

# Model definition
model = GCN_Net(data, args.n_hidden).to(device)
print("AC: model defined")

# Training stage

torch.manual_seed(args.random_seed)

optimizer_name = "Adam"
optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=args.learning_rate)

print("Training start.") 
summary_writer = SummaryWriter(args.output_directory) # Usage: tensorboard --logdir=./
for epoch in range(args.n_epochs):
    loss=train(model, data, optimizer)
    if epoch%args.log_freq==0: print(f"{epoch=} {loss=}")
    summary_writer.add_scalar('Loss',torch.FloatTensor([loss]),global_step=epoch)

print("Test start.")
train_acc,test_acc = testing_stage(model, data, 
                                   labels, 
                                   species, is_H, resnames, in_backbone, 
                                   args.classification, args.pdbid, args.output_directory, 
                                   inpname=args.input_directory+args.pdbid+'_protein.mol2')

print('=' * 50)
print('Train Accuracy: {} Test Accuray: {}'.format(train_acc,test_acc))
print('=' * 50)

