
import sys
sys.path.append('./src/')

import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split
import torch_geometric.transforms as T

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from functions import *

# Custom dataset
class MolDataset(InMemoryDataset):
    #def __init__(self, transform=None):
    def __init__(self, G, edge_index, embeddings, labels, test_size, random_seed, transform=None):
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
                                                            test_size=test_size, 
                                                            random_state=random_seed)
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


# GCN model
class GCN_Net(torch.nn.Module):
    def __init__(self, data, n_hidden):
        super(GCN_Net, self).__init__()
        self.data = data
        self.n_hidden = n_hidden
        self.conv1 = GCNConv(data.num_features, n_hidden)
        self.conv2 = GCNConv(n_hidden, n_hidden)
        self.conv3 = GCNConv(n_hidden, n_hidden)
        self.conv4 = GCNConv(n_hidden, int(data.num_classes))

    def forward(self):
        x, edge_index = self.data.x, self.data.edge_index
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
    
def train(model, data, optimizer):
    model.train()
    #loss = F.nll_loss(model()[data.train_mask], data.y[data.train_mask]) #negative log likelihood https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html
    loss = F.cross_entropy(model()[data.train_mask], data.y[data.train_mask]) # https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py (Link found in here: https://towardsdatascience.com/a-beginners-guide-to-graph-neural-networks-using-pytorch-geometric-part-1-d98dc93e7742)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach().item()    
   
@torch.no_grad()
def testing_stage(model, data,
                  labels, species, is_H, resnames, in_backbone, 
                  classification, pdbid, output_directory, 
                  inpname):
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

    results(mask, pred, indeces, labels, species, is_H, resnames, in_backbone, inpname)
    if classification=='is_H':
        for what_sel in ['species','resnames','in_backbone']:
            results_per_sel(pred, 
                            indeces, labels,
                            species, is_H, resnames, in_backbone, 
                            pdbid, 
                            output_directory, 
                            what_sel=what_sel)
    return acc1,acc
