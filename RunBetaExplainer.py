from ShapeGGenSimulator import ShapeGGen, return_ShapeGGen_dataset, graph_exp_acc, set_seed
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GATv2Conv, Linear, global_mean_pool, global_max_pool
import numpy as np
import pandas as pd
import torch
import torch_geometric
import optuna
import sys
import random
from torch_geometric.data import Data, DataLoader
import torch_geometric.transforms as T
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score, accuracy_score
from torch_geometric.utils import k_hop_subgraph, to_undirected
import igraph as ig
import networkx as nx
# Takes 4 arguments
# Argument 1: Denotes data type; specify 25 or 50 for the 25% and 50% sparse SERGIO datasets; base, hetero, unfair, lessinform, or moreinform for ShapeGGen Simulator datasets; specify file name for other files (assumed to be of format labels = f"Labels{file_name}', features = f"Features{file_name}', edge_index = f"EdgeIndex{file_name}' where file_name includes the file type (of csv, npy, tsv, or xlsx) consistant across all types, features is node/graph by number of features, labels if in a csv/tsv/excel is in column "Labels", and edge indices in a csv/tsv/excel is assumed to be in columns "P1" and "P2" denoting connectivity)
# Argument 2: denotes classification problem type; important for specified files
# Argument 3: denotes convolutional layer types chosen for model --> best parameters learned in this file
# Argument 4: denotes whether dataset has known groundtruth
def get_general_data(dta):
    ''' 
        This function loads a generic dataset with a file including labels caled Labels{dta}, feature file called Features{dta}, and edge Index file called EdgeIndex{dta}.
        It takes the dta where dta denotes specific labels for these sides and the file type - ie 1.npy would be associated with Labels1.npy, Features1.npy, and EdgeIndex1.npy.
        This takes into account input datasets.
    '''
    file_name = dta
    if file_name[-3:] == 'npy':
        labels = np.load('Labels' + file_name, allow_pickle=True)
        features = np.load('Features' + file_name, allow_pickle=True)
        adj = np.load('EdgeIndex' + file_name, allow_pickle=True)
    else:
        if file_name[-3:] == 'csv':
            df_labels = pd.read_csv('Labels' + file_name)
            df_feat = pd.read_csv('Features' + file_name)
            df_ei = pd.read_csv('EdgeIndex' + file_name)
        elif file_name[-3:] == 'tsv':
            df_labels = pd.read_csv('Labels' + file_name, sep='\t')
            df_feat = pd.read_csv('Features' + file_name, sep='\t')
            df_ei = pd.read_csv('EdgeIndex' + file_name, sep='\t')
        else:
            df_labels = pd.read_excel('Labels' + file_name)
            df_feat = pd.read_excel('Features' + file_name)
            df_ei = pd.read_excel('EdgeIndex' + file_name)
        labels = list(df_labels['Labels'])   
        features = df_feat.to_numpy()
        lst1 = list(df_ei['P1'])
        lst2 = list(df_ei['P2'])
        adj = np.array([lst1, lst2])
    num_features = features.shape[1]
    num_classes = len(np.unique(labels))
    return adj, features, labels, num_features, num_classes
def get_sergio_data(num):
    '''This function allows us to load our data. We use the supergraph data - the original graph plus
        extra points associated with differnet points that are highly correlated to see if our explainers
        capture the ground truth data well'''
    labels = np.load(f'Time Experiments/sergio data/SERGIOsimu_{num}Sparse_noLibEff_cTypes.npy')
    features = np.load(f'Time Experiments/sergio data/SERGIOsimu_{num}Sparse_noLibEff_concatShuffled.npy')
    num_features = features.shape[1]
    num_classes = len(np.unique(labels))
    adj = np.load(f'Time Experiments/sergio data/ExtraPointsSergio{num}.npy')
    gt = np.load('Time Experiments/sergio data/gt_adj.npy')
    lst1 = gt[0]
    lst2 = gt[1]
    gt = set()
    for i in range(0, len(lst1)):
        pt = (lst1[i], lst2[i])
        gt.add(pt)
    df_extra = np.load(f'Time Experiments/sergio data/ExtraPointsSergio{num}.npy')
    lst1 = df_extra[0]
    lst2 = df_extra[1]
    gt_grn = [] # Initialize list denoting whether edges in supergraph are in the original graph
    full_set = set()
    sz = len(lst1)
    for i in range(0, len(lst1)):
        pt = (lst1[i], lst2[i])
        full_set.add(pt)
        if pt in gt:
            gt_grn.append(1) # If in ground truth graph, add 1
        else:
            gt_grn.append(0) # Else add 0
    # The mask gt_grn indicates whether an edge in the edge index is in the groundtruth (denoted by 1) or not (denoted by 0)
    groundtruth_mask = torch.tensor(gt_grn)
    gt_grn = groundtruth_mask
    false_negative_base = 0
    l1 = np.load('Time Experiments/sergio data/gt_adj.npy')[0]
    l2 = np.load('Time Experiments/sergio data/gt_adj.npy')[1]
    for i in range(0, len(l1)):
        if (l1[i], l2[i]) not in full_set:
            false_negative_base += 1
    print(f'Number of Data-based FNs: {false_negative_base}')
    # false_negative_base are the edges in the groundtruth not present in the groundtruth as explainers cannot suggest these are present
    return adj, features, labels, num_features, num_classes, gt_grn, false_negative_base
def evaluate(y_pred, y_true):
    '''Return the accuracy, precision, recall, and F1 scores for GNN classification'''
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='micro')
    rec = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')
    return acc, prec, rec, f1 
def sergio_metrics(gt_grn, prediction_mask, false_negative_base):
    '''
        Return the accuracy, precision, recall, and F1 score for an explainer on the SERGIO groundtruth mask (gt_grn) given an explanation mask (prediction_mask).
        Also takes into account groundtruth elements not present in input edge index, which an explainer cannot predict upon.
    '''
    tp = 0
    tn = 0
    fn = false_negative_base
    fp = 0
    for ct in range(0, gt_grn.shape[0]):
        if gt_grn[ct] == 1 and prediction_mask[ct] > 0.5:
            tp += 1
        elif gt_grn[ct] == 1 and prediction_mask[ct] <= 0.5:
            fn += 1
        elif gt_grn[ct] == 0 and prediction_mask[ct] <= 0.5:
            tn += 1
        elif gt_grn[ct] == 0 and prediction_mask[ct] > 0.5:
            fp += 1
        else:
            continue
    acc = (tp + tn) / (tp + tn + fn + fp)
    if tp + fp != 0:
        prec = tp / (tp + fp)
    else:
        prec = 0
    if tp + fp != 0:
        rec = tp / (tp + fn)
    else:
        rec = 0
    if prec != 0 and rec != 0:
        f1 = (2 * prec * rec) / (prec + rec)
    else:
        f1 = 0
    return acc, prec, rec, f1
shapeggen = ['base', 'heterophilic', 'unfair', 'moreinform', 'lessinform', 'test']
sergio = ['25', '50']
conv_type = sys.argv[3]
# Denotes whether new user dataset has a groundtruth
if sys.argv[4] == 'True':
    groundtruth = True
else:
    groundtruth = False
if sys.argv[1] in shapeggen:
    '''Sets up dataset if it's a ShapeGGen dataset'''
    if sys.argv[1] == 'lessinform':
        num_features = 22
    else:
        num_features = 12
    x, y, edge_index, gt_exp, train_mask, test_mask, val_mask = return_ShapeGGen_dataset(sys.argv[1])
    num_classes = np.unique(y.numpy()).shape[0]
    num_hops = 3
elif sys.argv[1] in sergio:
    '''Sets up dataset if it's a SERGIO dataset'''
    adj, features, labels, num_features, num_classes, gt_grn, false_negative_base = get_sergio_data(sys.argv[1])
    edge_index = torch.tensor(adj, dtype=torch.int64)
    features = features.astype(np.float32)
    num_edges = np.array(adj).shape[1]
    edge_weight = torch.ones(num_edges)
    num_graphs = len(labels)
    graph_data = []
    shuffle_index = []
    for i in range(0, num_graphs):
        shuffle_index.append(i)
    shuffle_index = np.array(random.sample(shuffle_index, num_graphs))
    shuffle_index = shuffle_index.astype(np.int32)
    num_train = int(len(shuffle_index)* 0.8)
    num_test = num_graphs - num_train
    train_dataset = []
    test_dataset = []
    for j in range(0, num_graphs):
        i = shuffle_index[j]
        x = torch.tensor(features[i])
        y = torch.tensor(labels[i])
        data = Data(x=x, y=y, edge_index=edge_index)
        graph_data.append(data)
        if j < num_train:
            train_dataset.append(data)
        else:
            test_dataset.append(data)
    y = torch.tensor(labels)
    graph_data = torch_geometric.data.Batch.from_data_list(graph_data)
    graph_data = DataLoader(graph_data, batch_size=num_graphs)
    dataset = graph_data
    train_dataset = torch_geometric.data.Batch.from_data_list(train_dataset)
    test_dataset = torch_geometric.data.Batch.from_data_list(test_dataset)
    feat = torch.tensor(features)
    adjacency = torch.tensor(adj, dtype=torch.int64)
    y = torch.tensor(labels)
    train_loader = DataLoader(train_dataset, batch_size=num_train, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=num_test, shuffle=False)
elif sys.argv[1] == 'Texas':
    '''Sets up Texas dataset'''
    from torch_geometric.datasets import WebKB
    dataset = WebKB(root=f'/tmp/{sys.argv[1]}', name=sys.argv[1])
    data = dataset[0]
    x = data.x
    y = data.y
    edge_index = data.edge_index
    num_features = x.shape[1]
    num_classes = np.unique(y.numpy()).shape[0]
    num = y.shape[0]
    shuffle_index = []
    for i in range(0, num):
        shuffle_index.append(i)
    shuffle_index = np.array(random.sample(shuffle_index, num))
    shuffle_index = shuffle_index.astype(np.int32)
    train_mask = []
    train_idx = []
    test_mask = []
    test_idx = []
    val_mask = []
    val_idx = []
    num_train = int(len(shuffle_index)* 0.6)
    num_test = int(len(shuffle_index)*0.2)
    for j in range(0, num):
        i = shuffle_index[j]
        if j < num_train:
            train_idx.append(i)
        else:
            if j < num_train + num_test:
                test_idx.append(i)
            else:
                val_idx.append(i)
    for j in range(0, num):
        if j in train_idx:
            train_mask.append(True)
            test_mask.append(False)
            val_mask.append(False)
        elif j in test_idx:
            train_mask.append(False)
            test_mask.append(True)
            val_mask.append(False)
        else:
            train_mask.append(False)
            test_mask.append(False)
            val_mask.append(True)
    train_mask = torch.tensor(train_mask)
    test_mask = torch.tensor(test_mask)
    val_mask = torch.tensor(val_mask)
else:    
    '''Sets up general dataset'''
    adj, features, labels, num_features, num_classes = get_general_data(sys.argv[1])
    edge_index = torch.tensor(adj, dtype=torch.int64)
    features = features.astype(np.float32)
    num_edges = np.array(adj).shape[1]
    labels = labels.astype(np.int64)
    if groundtruth:
        '''Sets up Edge Index groundtruth mask to judge accuracy if relevent to generic dataset'''
        file_name = sys.argv[1]
        if file_name[-3:] == 'npy':
            gt = np.load('GTEdgeIndex' + file_name, allow_pickle=True).astype(np.int64)
            gt_set = set()
            gt_exp = []
            for i in range(0, gt.shape[1]):
                p1 = gt[0, i]
                p2 = gt[1, i]
                gt_set.add((p1, p2))
            ei = edge_index.numpy()
            for i in range(0, edge_index.shape[1]):
                p1 = ei[0, i]
                p2 = ei[1, i]
                if (p1, p2) in gt_set:
                    gt_exp.append(1)
                else:
                    gt_exp.append(0)
            gt_exp = torch.tensor(np.array(gt_exp))
        else:
            if file_name[-3:] == 'csv':
                df_ei = pd.read_csv('GTEdgeIndex' + file_name)
            elif file_name[-3:] == 'tsv':
                df_ei = pd.read_csv('GTEdgeIndex' + file_name, sep='\t')
            else:
                df_ei = pd.read_excel('GTEdgeIndex' + file_name)
            lst1 = list(df_gt['P1'])
            lst2 = list(df_gt['P2'])
            gt_exp = torch.tensor(np.array([lst1, lst2]).astype(np.int64))
    if sys.argv[2] == 'node':
        '''Sets up dataset if it is a node classification problem'''
        x = torch.tensor(features)
        y = torch.tensor(labels)
        num_nodes = features.shape[0]
        shuffle_index = []
        for i in range(0, num_nodes):
            shuffle_index.append(i)
        shuffle_index = np.array(random.sample(shuffle_index, num_nodes))
        shuffle_index = shuffle_index.astype(np.int32)
        num_train = int(len(shuffle_index)* 0.6)
        num_test = int(len(shuffle_index)* 0.2)
        train_idx = []
        test_idx = []
        val_idx = []
        for j in range(0, num_nodes):
            i = shuffle_index[j]
            if j < num_train:
                train_idx.append(i)
            else:
                if j < num_train + num_test:
                    test_idx.append(i)
                else:
                    val_idx.append(i)
        train_mask = []
        test_mask = []
        val_mask = []
        for j in range(0, num_nodes):
            if j in train_idx:
                train_mask.append(True)
                test_mask.append(False)
                val_mask.append(False)
            elif j in test_idx:
                train_mask.append(False)
                test_mask.append(True)
                val_mask.append(False)
            else:
                train_mask.append(False)
                test_mask.append(False)
                val_mask.append(True)
        train_mask = torch.tensor(train_mask)
        test_mask = torch.tensor(test_mask)
        val_mask = torch.tensor(val_mask)
    else:
        ''Sets up dataset if it is a graph classification problem'''
        edge_weight = torch.ones(num_edges)
        num_graphs = len(labels)
        graph_data = []
        shuffle_index = []
        for i in range(0, num_graphs):
            shuffle_index.append(i)
        shuffle_index = np.array(random.sample(shuffle_index, num_graphs))
        shuffle_index = shuffle_index.astype(np.int32)
        num_train = int(len(shuffle_index)* 0.8)
        num_test = num_graphs - num_train
        train_dataset = []
        test_dataset = []
        for j in range(0, num_graphs):
            i = shuffle_index[j]
            x = torch.tensor(features[i])
            y = torch.tensor(labels[i])
            data = Data(x=x, y=y, edge_index=edge_index)
            graph_data.append(data)
            if j < num_train:
                train_dataset.append(data)
            else:
                test_dataset.append(data)
        y = torch.tensor(labels)
        graph_data = torch_geometric.data.Batch.from_data_list(graph_data)
        graph_data = DataLoader(graph_data, batch_size=num_graphs)
        print(graph_data)
        train_dataset = torch_geometric.data.Batch.from_data_list(train_dataset)
        test_dataset = torch_geometric.data.Batch.from_data_list(test_dataset)
        feat = torch.tensor(features)
        adjacency = torch.tensor(adj, dtype=torch.int64)
        y = torch.tensor(labels)
        train_loader = DataLoader(train_dataset, batch_size=num_train, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=num_test, shuffle=False)
# Sets up input features based on classification problem
if sys.argv[2] == 'graph':
    input_features = 1
else:
    input_features = num_features
device = torch.device('cpu')
# Import BetaExplainer files for running BetaExplainer
import NodeBetaExplainer, GraphBetaExplainer
# Determine whether model incorporates final linear layer (if GATConv and GATv2Conv layers are not used
if sys.argv[5] == 'True':
    lin = True
else:
    lin = False
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, output_size, num_layers, conv_type, heads):
        super(GCN, self).__init__()
        ''' Define the general GNN model based on a given convolution type (conv_type), hidden channels, number of layers, and heads if it's an attention based model '''
        self.gat_layers = torch.nn.ModuleList()
        for i in range(0, num_layers):
            if i == 0:
                if conv_type == 'GAT':
                    conv = GATConv(num_features, hidden_channels, heads)
                elif conv_type == 'GATv2':
                    conv = GATv2Conv(num_features, hidden_channels, heads)
                elif conv_type == 'GCN':
                    if not lin and num_layers == 1:
                        conv = GCNConv(num_features, output_size)
                    else:
                        conv = GCNConv(num_features, hidden_channels)
                else:
                    if not lin and num_layers == 1:
                        conv = SAGEConv(num_features, output_size)
                    else:
                        conv = SAGEConv(num_features, hidden_channels)
            else:   
                if conv_type == 'GAT':
                     conv = GATConv(hidden_channels * heads, hidden_channels, heads)
                elif conv_type == 'GATv2':
                    conv = GATv2Conv(hidden_channels * heads, hidden_channels, heads)
                elif conv_type == 'GCN':
                    if not lin and i == num_layers - 1:
                        conv = GCNConv(hidden_channels, output_size)
                    else:
                        conv = GCNConv(hidden_channels, hidden_channels)
                else:
                    if not lin and i == num_layers - 1:
                        conv = SAGEConv(hidden_channels, output_size)
                    else:
                        conv = SAGEConv(hidden_channels, hidden_channels)
            self.gat_layers.append(conv)
        self.num_layers = num_layers
        if conv_type in ['GAT', 'GATv2']:
            i = hidden_channels * heads
            self.lin = Linear(i, output_size)
        else:
            if lin:
                i = hidden_channels
                self.lin = Linear(i, output_size)
        self.conv_type = conv_type
    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None: # No batch given
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            batch = torch.zeros(x.size(0), dtype=torch.long).to(device)
        for lay in range(0, self.num_layers):
            if self.conv_type in ['GAT', 'GATv2']:
                x = self.gat_layers[lay](x, edge_index, return_attention_weights=True)
                x = x[0]
            else:
                x = self.gat_layers[lay](x, edge_index)
        x = F.dropout(x, p=0.2, training=self.training)
        if lin:
            x = self.lin(x)
        if sys.argv[2] == 'graph':
            x = global_max_pool(x, batch)
        return x
y_true = y.numpy()
criterion = torch.nn.CrossEntropyLoss()
def model_objective(trial):
    # Find the best parameters for a model on the given dataset
    lr = trial.suggest_float('lrs', 1e-6, 0.2)
    wd = trial.suggest_float('wds', 0, 1)
    hcs = trial.suggest_int('hcs', 2, 512)
    layers = trial.suggest_int('layers', 1, 5)
    if conv_type in ['GAT', 'GATv2']:
        upper = 5
    else:
        upper = 0
    heads = trial.suggest_int('heads', 0, upper)
    model = GCN(input_features, hcs, num_classes, layers, conv_type, heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    num_epochs = 5
    if sys.argv[2] == 'node':
        # train if it's a node classification problem
        y_true = y.numpy()
        for epoch in range(1, num_epochs + 1):
            model.train()
            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = criterion(out[train_mask], y[train_mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)
            optimizer.step()
            model.eval()
            y_pred = model(x, edge_index).argmax(axis=1).numpy()
            train_acc, prec, rec, f1 = evaluate(y_pred[train_mask], y_true[train_mask])
            test_acc, prec, rec, f1 = evaluate(y_pred[test_mask], y_true[test_mask])
            val_acc, prec, rec, f1 = evaluate(y_pred[val_mask], y_true[val_mask])
            # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Val Acc: {val_acc:.4f}, Loss: {loss:.4f}')
    else:
        # train if it's a graph classification problem
        for epoch in range(1, num_epochs + 1):
            model.train()
            avgLoss = 0
            for data in tqdm(train_loader, total=47):  # Iterate in batches over the training dataset.
                data.x = torch.reshape(data.x, (data.x.shape[0], 1))
                data.x = data.x.type(torch.FloatTensor)
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)# Perform a single forward pass
                loss = criterion(out, data.y)  # Compute the loss.
                loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
                optimizer.zero_grad()  # Clear gradients.
                avgLoss += loss
            loss = avgLoss / 47
            model.eval()
            correct = 0
            for data in train_loader:  # Iterate in batches over the training/test dataset.
                data.x = torch.reshape(data.x, (data.x.shape[0], 1))
                data.x = data.x.type(torch.FloatTensor)
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)  
                pred = out.argmax(dim=1)  # Use the class with highest probability.
                correct += int((pred == data.y).sum())  # Check against ground-truth labels.
            train_acc = correct / len(train_loader.dataset)
            correct = 0
            for data in test_loader:  # Iterate in batches over the training/test dataset.
                data.x = torch.reshape(data.x, (data.x.shape[0], 1))
                data.x = data.x.type(torch.FloatTensor)
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)  
                pred = out.argmax(dim=1)  # Use the class with highest probability.
                correct += int((pred == data.y).sum())  # Check against ground-truth labels.
            test_acc = correct / len(test_loader.dataset)
            # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Loss: {loss:.4f}')
    return test_acc
pruner = optuna.pruners.MedianPruner()
# get best parameters for the chosen GNN
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), pruner=pruner)
n_tr = 500
study.optimize(model_objective, n_trials=n_tr)
# get GNN training epochs based on dataset
if sys.argv[1] in shapeggen:
    num_epochs = 2000
elif sys.argv[1] == 'Texas':
    num_epochs = 250
else:
    num_epochs = 50
# Get Best Parameters for dataset
print('Best hyperparameters:', study.best_params)
print('Best Result:', study.best_value)
lr = study.best_params['lrs']
wd = study.best_params['wds']
hcs = study.best_params['hcs']
layers = study.best_params['layers']
heads = study.best_params['heads']
criterion = torch.nn.CrossEntropyLoss()
model = GCN(input_features, hcs, num_classes, layers, conv_type, heads).to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
# Train model based on classification problem type
if sys.argv[2] == 'node':
    y_true = y.numpy()
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)
        optimizer.step()
        model.eval()
        y_pred = model(x, edge_index).argmax(axis=1).numpy()
        train_acc, train_prec, train_rec, train_f1 = evaluate(y_pred[train_mask], y_true[train_mask])
        test_acc, test_prec, test_rec, test_f1  = evaluate(y_pred[test_mask], y_true[test_mask])
        val_acc, val_prec, val_rec, val_f1  = evaluate(y_pred[val_mask], y_true[val_mask])
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc}, Test Acc: {test_acc}, Val Acc: {val_acc}, Loss: {loss}')
else:
    for epoch in range(1, num_epochs + 1):
        model.train()
        avgLoss = 0
        for data in tqdm(train_loader, total=47):  # Iterate in batches over the training dataset.
            data.x = torch.reshape(data.x, (data.x.shape[0], 1))
            data.x = data.x.type(torch.FloatTensor)
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)# Perform a single forward pass
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
            avgLoss += loss
        loss = avgLoss / 47
        model.eval()
        correct = 0
        for data in train_loader:  # Iterate in batches over the training/test dataset.
            data.x = torch.reshape(data.x, (data.x.shape[0], 1))
            data.x = data.x.type(torch.FloatTensor)
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)  
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        train_acc = correct / len(train_loader.dataset)
        correct = 0
        for data in test_loader:  # Iterate in batches over the training/test dataset.
            data.x = torch.reshape(data.x, (data.x.shape[0], 1))
            data.x = data.x.type(torch.FloatTensor)
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)  
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        test_acc = correct / len(test_loader.dataset)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc}, Test Acc: {test_acc}, Loss: {loss}')

def faithfulness(model, X, G, edge_mask):
    ''' This function defines the similarity of the model output on the edge index masked out by the edge mask to the original output for node classification datasets '''
    org_vec = model(X, G)
    lst = []
    for i in range(0, edge_mask.shape[0]):
        if edge_mask[i] >= 0.5:
            lst.append(i)
    g = G[:, lst]
    pert_vec = model(X, g)
    org_softmax = F.softmax(org_vec, dim=-1)
    pert_softmax = F.softmax(pert_vec, dim=-1)
    res = 1 - torch.exp(-F.kl_div(org_softmax.log(), pert_softmax, None, None, 'sum')).item()
    return res

def graph_faithfulness(model, X, G, edge_mask):
    ''' This function defines the similarity of the model output on the edge index masked out by the edge mask to the original output for graph classification datasets '''
    org_vec = []
    for data in X:
        data.x = torch.reshape(data.x, (data.x.shape[0], 1))
        data.x = data.x.type(torch.FloatTensor)
        data = data.to(device)
        org_vec1 = model(data.x, G, data.batch).tolist()
        org_vec.append(org_vec1)
    org_vec = torch.tensor(org_vec)
    lst = []
    for i in range(0, edge_mask.shape[0]):
        if edge_mask[i] >= 0.5:
            lst.append(i)
    g = G[:, lst]
    
    pert_vec = []
    for data in X:
        data.x = torch.reshape(data.x, (data.x.shape[0], 1))
        data.x = data.x.type(torch.FloatTensor)
        data = data.to(device)
        pert_vec1 = model(data.x, g, data.batch).tolist()
        pert_vec.append(pert_vec1)
    pert_vec = torch.tensor(pert_vec)
    
    org_softmax = F.softmax(org_vec, dim=-1)
    pert_softmax = F.softmax(pert_vec, dim=-1)
    res = 1 - torch.exp(-F.kl_div(org_softmax.log(), pert_softmax, None, None, 'sum')).item()
    return res

ei = edge_index

def shapeggen_objective(trial):
    ''' Get the best hyperparameters for the ShapeGGen dataset to mimic the methods of measuring explanation accuracy '''
    lr = trial.suggest_float('lrs', 1e-6, 0.1)
    alpha = trial.suggest_float('a', 0.1, 1)
    beta = trial.suggest_float('b', 0.1, 1)
    explainer = NodeBetaExplainer.BetaExplainer(model, x, edge_index, torch.device('cpu'), alpha, beta)
    explainer.train(5, lr)
    betaem = explainer.edge_mask()
    best_acc = 0
    for i in range(0, len(gt_exp)):
        subset, sub_edge_index, mapping, hard_edge_mask = \
            k_hop_subgraph(i, num_hops, edge_index,
                          relabel_nodes=False)
        ei = edge_index[:,hard_edge_mask]
        exp = betaem[hard_edge_mask]
        accuracy, f1, prec, rec = graph_exp_acc(gt_exp[i], exp, node_thresh_factor = 0.5)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_faith = faithfulness(model, x, ei, exp)
    best_result = (best_acc + 1 - best_faith) / 2 # get parameters associated with the best accuracy and faithfulness
    return best_result

def sergio_objective(trial):
    ''' Get the best hyperparameters for the SERGIO dataset '''
    lr = trial.suggest_float('lrs', 1e-6, 0.1)
    alpha = trial.suggest_float('a', 0.1, 1)
    beta = trial.suggest_float('b', 0.1, 1)
    explainer = GraphBetaExplainer.BetaExplainer(model, graph_data, edge_index, torch.device('cpu'), 2000, alpha, beta)
    explainer.train(5, lr)
    prediction_mask = explainer.edge_mask()
    em = prediction_mask
    acc, prec, rec, f1 = sergio_metrics(gt_grn, prediction_mask, false_negative_base) # get parameters associated with the best accuracy
    return acc
# denote whether a new, non-paper dataset has a notion of groundtruth
gt = sys.argv[4]
if gt == 'True':
    groundtruth = True
else:
    groundtruth = False

def exp_acc(gt_exp, betaem):
    ''' Calculate explanation accuracy for a general dataset ''''
    gt = gt_exp.numpy()
    em = betaem.numpy()
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(0, gt.shape[0]):
        if gt[i] == 1:
            if em[i] >= 0.5:
                tp += 1
            else:
                fn += 1
        else:
            if em[i] >= 0.5:
                fp += 1
            else:
                tn += 1
    acc = (tp + tn) / (tp + tn + fp + fn)
    denom = tp + fp
    if denom != 0:
        prec = tp / denom
    else:
        prec = 0
    denom = tp + fn
    if denom != 0:
        rec = tp / denom
    else:
        rec = 0
    if prec == 0 and rec == 0:
        f1 = 0
    else:
        f1 = (2 * prec * rec) / (prec + rec)
    return acc, prec, rec, f1

def node_objective(trial):
    ''' Get best parameters for general node classification GNN problem '''
    lr = trial.suggest_float('lrs', 1e-6, 0.05)
    alpha = trial.suggest_float('a', 0.1, 5)
    beta = trial.suggest_float('b', 0.1, 5)
    explainer = NodeBetaExplainer.BetaExplainer(model, x, edge_index, torch.device('cpu'), alpha, beta)
    explainer.train(5, lr)
    betaem = explainer.edge_mask()
    faith = faithfulness(model, x, edge_index, betaem) # faithfulness may be measured for all datasets so is incorporated as part of metric
    res = 1 - faith
    if groundtruth:
        accuracy, prec, rec, f1 = exp_acc(gt_exp, betaem)
        res += accuracy
        res /= 2 # averages 1 - faithfulness and accuracy for groundtruth datasets (we want to increase both values)
    else:
        # if no groundtruth, ensure explanation is sparse (IE doesn't include too many edges)
        em = betaem.numpy()
        sparse = 0
        for i in range(0, em.shape[0]):
            if em[i] >= 0.5:
                sparse += 1
        calc = sparse / em.shape[0]
        if calc > 0 and calc <= 0.6 and faith < 0.95:
            res += 1 - calc
            res /= 2
        else:
            res = 0
    return res

def graph_objective(trial):
    ''' Get best parameters for general graph classification GNN problem '''
    lr = trial.suggest_float('lrs', 1e-6, 0.05)
    alpha = trial.suggest_float('a', 0.1, 5)
    beta = trial.suggest_float('b', 0.1, 5)
    explainer = GraphBetaExplainer.BetaExplainer(model, graph_data, edge_index, torch.device('cpu'), num_graphs, alpha, beta)
    explainer.train(5, lr)
    betaem = explainer.edge_mask()
    faith = graph_faithfulness(model, graph_data, edge_index, betaem) # faithfulness may be measured for all datasets so is incorporated as part of metric
    res = 1 - faith
    if groundtruth:
        accuracy, prec, rec, f1 = exp_acc(gt_exp, betaem)
        res += accuracy
        res /= 2 # averages 1 - faithfulness and accuracy for groundtruth datasets (we want to increase both values)
    else:
        # if no groundtruth, ensure explanation is sparse (IE doesn't include too many edges)
        em = betaem.numpy()
        sparse = 0
        for i in range(0, em.shape[0]):
            if em[i] >= 0.5:
                sparse += 1
        calc = sparse / em.shape[0]
        if calc > 0 and calc <= 0.6 and faith < 0.95:
            res += 1 - calc
            res /= 2
        else:
            res = 0
    return res
# get best BetaExplainer parameters    
pruner = optuna.pruners.MedianPruner()
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), pruner=pruner)
if sys.argv[1] in shapeggen:
    study.optimize(shapeggen_objective, n_trials=n_tr)
elif sys.argv[1] in sergio:
    study.optimize(sergio_objective, n_trials=n_tr)
else:
    if sys.argv[2] == 'node':
        study.optimize(node_objective, n_trials=n_tr)
    else:
        study.optimize(graph_objective, n_trials=n_tr)
print('Best hyperparameters:', study.best_params)
print('Best Result:', study.best_value)

lr = study.best_params['lrs']
alpha = study.best_params['a']
beta = study.best_params['b']
ep = 500
results = []
graphs = []
runs = 10
# run 10 runs of BetaExplainer with random seeds each time, to see variability of the chosen explanation
# add results and record probabilities for each edge for all graphs
for run in range(0, runs):
    seed = np.random.randint(0, 1000001)
    set_seed(int(seed))
    if sys.argv[1] in shapeggen:
        explainer = NodeBetaExplainer.BetaExplainer(model, x, edge_index, torch.device('cpu'), alpha, beta)
        explainer.train(ep, lr)
        betaem = explainer.edge_mask()
        best_acc = 0
        for i in range(0, len(gt_exp)):
            subset, sub_edge_index, mapping, hard_edge_mask = \
                k_hop_subgraph(i, num_hops, edge_index,
                              relabel_nodes=False)
            ei = edge_index[:,hard_edge_mask]
            exp = betaem[hard_edge_mask]
            accuracy, prec, rec, f1 = graph_exp_acc(gt_exp[i], exp, node_thresh_factor = 0.5)
            if accuracy >= best_acc:
                best_acc = accuracy
                best_f1 = f1
                best_prec = prec
                best_rec = rec
                best_faith = faithfulness(model, x, ei, exp)
                best_exp = exp.numpy()
                best_ei = ei.numpy()
                best_gt = gt_exp[i].edge_imp.numpy()
        print(f'Best Accuracy: {best_acc}, Best Precision: {best_prec}, Best Recall: {best_rec}, Best F1 Score: {best_f1}., Best Unfaithfulness: {best_faith}')
        out = [seed, best_acc, best_prec, best_rec, best_f1, best_faith]
        for i in range(0, best_exp.shape[0]):
            graphs.append([seed, best_exp[i], best_ei[0, i], best_ei[1, i], best_gt[i]])
    elif sys.argv[1] in sergio:
        explainer = GraphBetaExplainer.BetaExplainer(model, graph_data, edge_index, torch.device('cpu'), 2000, alpha, beta)
        explainer.train(ep, lr)
        prediction_mask = explainer.edge_mask()
        em = prediction_mask
        acc, prec, rec, f1 = sergio_metrics(gt_grn, prediction_mask, false_negative_base)
        faith = graph_faithfulness(model, graph_data, edge_index, em)
        print(f'Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1 Score: {f1}, Unfaithfulness: {faith}')
        out = [seed, acc, prec, rec, f1, faith]
        em = em.numpy()
        ei = edge_index.numpy()
        gt = gt_grn.numpy()
        for i in range(0, em.shape[0]):
            graphs.append([seed, em[i], ei[0, i], ei[1, i], gt[i]])
    else:
        if sys.argv[2] == 'node':
            explainer = NodeBetaExplainer.BetaExplainer(model, x, edge_index, torch.device('cpu'), alpha, beta)
            explainer.train(ep, lr)
            betaem = explainer.edge_mask()
            faith = faithfulness(model, x, edge_index, betaem)
        else:
            explainer = GraphBetaExplainer.BetaExplainer(model, graph_data, edge_index, torch.device('cpu'), num_graphs, alpha, beta)
            explainer.train(ep, lr)
            betaem = explainer.edge_mask()
            faith = graph_faithfulness(model, graph_data, edge_index, betaem)
        if groundtruth:
            accuracy, prec, rec, f1 = exp_acc(gt_exp, betaem)
            print(f'Accuracy: {accuracy}, Precision: {prec}, Recall: {rec}, F1 Score: {f1}, Unfaithfulness: {faith}')
            out = [seed, accuracy, prec, rec, f1, faith]
            em = betaem.numpy()
            ei = edge_index.numpy()
            gt = gt_exp.numpy()
            for i in range(0, em.shape[0]):
                graphs.append([seed, em[i], ei[0, i], ei[1, i], gt[i]])
        else:
            em = betaem.numpy()
            sparse = 0
            for i in range(0, em.shape[0]):
                if em[i] >= 0.5:
                    sparse += 1
            sparse /= em.shape[0]
            print(f'Unfaithfulness: {faith}, Kept Edges: {sparse}')
            out = [seed, faith, sparse]
            ei = edge_index.numpy()
            for i in range(0, em.shape[0]):
                graphs.append([seed, em[i], ei[0, i], ei[1, i]])
    results.append(out)
# generate returned csv based on the chosen metrics (taking into account presence or absence of groundtruth
if len(out) > 3:
    cols = ['Seed', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Unfaithfulness']
    cols1 = ['Seed', 'Probability', 'P1', 'P2', 'Groundtruth']
else:
    cols = ['Seed', 'Unfaithfulness', 'Kept Edges']
    cols1 = ['Seed', 'Probability', 'P1', 'P2']
df = pd.DataFrame(results, columns=cols)
fn = sys.argv[1]
if fn != 'Texas' and fn not in sergio and fn not in shapeggen:
    fn = fn[0:-4]
df.to_csv(f'SeedResults{fn}.csv')
df1 = pd.DataFrame(graphs, columns=cols1)
df1.to_csv(f'SeedGraphResults{fn}.csv')
# Return average metrics for datasets taking into account the Texas dataset does not have a notion of groundtruth --> only faithfulness and number of edge indices kept
# Return the seed for the best performing metrics for each explainer to choose this seed as to chose this explanation to graph
if 'Accuracy' in cols:
    a = list(df['Accuracy'])
    acc = np.mean(a)
    p = list(df['Precision'])
    prec = np.mean(p)
    r = list(df['Recall'])
    rec = np.mean(r)
    fs = list(df['F1 Score'])
    f1 = np.mean(fs)
    un = list(df['Unfaithfulness'])
    unfaith = np.mean(un)
    print(f'Average Accuracy: {acc}, Average Precision: {prec}, Average Recall: {rec}, Average F1 Score: {f1}, Average Unfaithfulness: {unfaith}')
    best_acc = 0
    best_f1 = 0
    best_faith = 1
    idx = 0
    for i in range(0, len(a)):
        if a[i] >= best_acc and fs[i] >= best_f1 and un[i] <= best_faith:
            idx = i
            best_acc = a[i]
            best_f1 = fs[i]
            best_faith = un[i]
else:
    f = list(df['Unfaithfulness'])
    unfaith = np.mean(f)
    k = list(df['Kept Edges'])
    sparse = np.mean(k)
    print(f'Average Unfaithfulness: {unfaith}, Average Fraction of Kept Edges: {sparse}')
    best_faith = 1
    best_sparse = 1
    idx = 0
    for i in range(0, len(f)):
        if f[i] <= best_faith and k[i] <= best_sparse:
            idx = i
            best_faith = f[i]
            best_sparse = k[i]
best_seed = list(df['Seed'])[idx]
df1 = df1[df1['Seed'] == best_seed]
# graph best BetaExplanation explanation graph
G = nx.DiGraph() 
if sys.argv[1] == 'Texas':
    pos = df1[df1['Probability'] >= 0.5]
    pos_set = []
    p1s = list(pos['P1'])
    p2s = list(pos['P2'])
    for i in range(len(p1s)):
        p1 = p1s[i]
        p2 = p2s[i]
        if p1 not in pos_set:
            pos_set.append(p1)
        if p2 not in pos_set:
            pos_set.append(p2)
    actual = y.numpy()
    color = dict()
    color[0] = '#2E2585'
    color[1] = '#337538'
    color[2] = '#5DA899'
    color[3] = '#94CBEC'
    color[4] = '#DCCD7D'
    for node in pos_set:
        # add nodes associated with edge included in explanation graph (IE probability of at least 0.5) and denote the class through color for Texas
        # this will give a sense of whether the explanation graph is more homophilic than the original Texas graph
        col = color[actual[node]]
        G.add_node(node, color=col)
lst = []
weights = []
probs = list(df1['Probability'])
mx = np.max(probs)
mn = np.min(probs)
if sys.argv[1] in shapeggen or sys.argv[1] in sergio or groundtruth:
    # ShapeGGen, SERGIO, and specified new datasets with groundtruth have a notion of groundtruth
    # we make a set of the groundtruth edges tp_set to check edges against
    tp_edges = df1[df1['Groundtruth'] == 1]
    tp_set = set()
    p1s = list(tp_edges['P1'])
    p2s = list(tp_edges['P2'])
    for i in range(len(p1s)):
        p1 = p1s[i]
        p2 = p2s[i]
        tp_set.add((p1, p2))
    true_edge = '#1A85FF'
    false_positive_edge = '#D41159'
    false_negative_edge = '#ED9FBC'
b1 = list(df1['P1'])
b2 = list(df1['P2'])
for i in range(0, len(b1)):
    p1 = b1[i]
    p2 = b2[i]
    if probs[i] >= 0.5:
        # add chosen edges --> those with probability of at least 0.5
        if sys.argv[1] in shapeggen or sys.argv[1] in sergio or groundtruth:
            if (p1, p2) in tp_set:
                color = true_edge # if it's an edge in the groundtruth, denote it as a blue color
            else:
                color = false_positive_edge # if it's an edge not in the groundtruth, denote it as a magenta edge to indicate it's inaccurate
            G.add_edge(p1, p2, color=color)
        else:
            G.add_edge(p1, p2)  # no edge color added for datasets without a notion of groundtruth
        p = (probs[i] - mn + 1e-5) / (mx - mn) # add edge weighting based on edge probability
        # p = probs[i]
        weights.append(5 * p)
    else:
        # for datasets with groundtruth, add edges that are false negatives, labeling them with light pink
        # this will indicate that the edge should be in the explanation but isn't
        if sys.argv[1] in shapeggen or sys.argv[1] in sergio or groundtruth:
            if (p1, p2) in tp_set:
                G.add_edge(p1, p2, color=false_negative_edge)
h = ig.Graph.from_networkx(G)
ig.plot(h, vertex_size=7, edge_width=weights, target=f'{fn}Plot.png')
