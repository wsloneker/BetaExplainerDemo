from ShapeGGenSimulator import ShapeGGen, return_ShapeGGen_dataset, graph_exp_acc
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
# Takes 4 arguments
# Argument 1: Denotes data type; specify 25 or 50 for the 25% and 50% sparse SERGIO datasets; base, hetero, unfair, lessinform, or more inform for ShapeGGen Simulator datasets; specify file name for other files (assumed to be of format labels = f"Labels{file_name}', features = f"Features{file_name}', edge_index = f"EdgeIndex{file_name}' where file_name includes the file type (of csv, npy, tsv, or xlsx) consistant across all types, features is node/graph by number of features, labels if in a csv/tsv/excel is in column "Labels", and edge indices in a csv/tsv/excel is assumed to be in columns "P1" and "P2" denoting connectivity)
# Argument 2: denotes classification problem type; important for specified files
# Argument 3: denotes convolutional layer types chosen for model --> best parameters learned in this file
# Argument 4: denotes whether dataset has known groundtruth
def get_general_data(dta):
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
    groundtruth_mask = torch.tensor(gt_grn)
    gt_grn = groundtruth_mask
    false_negative_base = 0
    l1 = np.load('Time Experiments/sergio data/gt_adj.npy')[0]
    l2 = np.load('Time Experiments/sergio data/gt_adj.npy')[1]
    for i in range(0, len(l1)):
        if (l1[i], l2[i]) not in full_set:
            false_negative_base += 1
    print(f'Number of Data-based FNs: {false_negative_base}')
    return adj, features, labels, num_features, num_classes, gt_grn, false_negative_base
def evaluate(y_pred, y_true):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='micro')
    rec = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')
    return acc, prec, rec, f1 
def sergio_metrics(gt_grn, prediction_mask, false_negative_base):
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
if sys.argv[4] == 'True':
    groundtruth = True
else:
    groundtruth = False
if sys.argv[1] in shapeggen:
    if sys.argv[1] == 'lessinform':
        num_features = 22
    else:
        num_features = 12
    x, y, edge_index, gt_exp, train_mask, test_mask, val_mask = return_ShapeGGen_dataset(sys.argv[1])
    num_classes = np.unique(y.numpy()).shape[0]
    num_hops = 3
elif sys.argv[1] in sergio:
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
else:    
    adj, features, labels, num_features, num_classes = get_general_data(sys.argv[1])
    edge_index = torch.tensor(adj, dtype=torch.int64)
    features = features.astype(np.float32)
    num_edges = np.array(adj).shape[1]
    labels = labels.astype(np.int64)
    if groundtruth:
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
if sys.argv[2] == 'graph':
    input_features = 1
else:
    input_features = num_features
device = torch.device('cpu')
import NodeBetaExplainer, GraphBetaExplainer
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, output_size, num_layers, conv_type, heads):
        super(GCN, self).__init__()
        self.gat_layers = torch.nn.ModuleList()
        for i in range(0, num_layers):
            if i == 0:
                if conv_type == 'GAT':
                    conv = GATConv(num_features, hidden_channels, heads)
                elif conv_type == 'GATv2':
                    conv = GATv2Conv(num_features, hidden_channels, heads)
                elif conv_type == 'GCN':
                    conv = GCNConv(num_features, hidden_channels)
                else:
                    conv = SAGEConv(num_features, hidden_channels)
            else:
                if conv_type == 'GAT':
                     conv = GATConv(hidden_channels * heads, hidden_channels, heads)
                elif conv_type == 'GATv2':
                    conv = GATv2Conv(hidden_channels * heads, hidden_channels, heads)
                elif conv_type == 'GCN':
                    conv = GCNConv(hidden_channels, hidden_channels)
                else:
                    conv = SAGEConv(hidden_channels, hidden_channels)
            self.gat_layers.append(conv)
        self.num_layers = num_layers
        if conv_type in ['GAT', 'GATv2']:
            i = hidden_channels * heads
        else:
            i = hidden_channels
        self.lin = Linear(i, num_classes)
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
        x = self.lin(x)
        if sys.argv[2] == 'graph':
            x = global_max_pool(x, batch)
        return x
y_true = y.numpy()
def model_objective(trial):
    lr = trial.suggest_float('lrs', 1e-6, 0.2)
    wd = trial.suggest_float('wds', 0, 1)
    hcs = trial.suggest_int('hcs', 2, 512)
    layers = trial.suggest_int('layers', 1, 5)
    if conv_type in ['GAT', 'GATv2']:
        upper = 5
    else:
        upper = 1
    heads = trial.suggest_int('heads', 0, upper)
    criterion = torch.nn.CrossEntropyLoss()
    model = GCN(input_features, hcs, num_classes, layers, conv_type, heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    num_epochs = 5
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
            train_acc, prec, rec, f1 = evaluate(y_pred[train_mask], y_true[train_mask])
            test_acc, prec, rec, f1 = evaluate(y_pred[test_mask], y_true[test_mask])
            val_acc, prec, rec, f1 = evaluate(y_pred[val_mask], y_true[val_mask])
            # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Val Acc: {val_acc:.4f}, Loss: {loss:.4f}')
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
            # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Loss: {loss:.4f}')
    return test_acc
pruner = optuna.pruners.MedianPruner()
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), pruner=pruner)
study.optimize(model_objective, n_trials=150)
if sys.argv[1] in shapeggen:
    num_epochs = 2000
else:
    num_epochs = 250
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
if sys.argv[2] == 'node':
    y_true = y.numpy()
    for epoch in range(1, num_epochs):
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
    for epoch in range(1, num_epochs):
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
            best_faith = faithfulness(model, x, edge_index, exp)
    best_result = (best_acc + 1 - best_faith) / 2
    return best_result

def sergio_objective(trial):
    lr = trial.suggest_float('lrs', 1e-6, 0.1)
    alpha = trial.suggest_float('a', 0.1, 1)
    beta = trial.suggest_float('b', 0.1, 1)
    explainer = GraphBetaExplainer.BetaExplainer(model, graph_data, edge_index, torch.device('cpu'), 2000, alpha, beta)
    explainer.train(5, lr)
    prediction_mask = explainer.edge_mask()
    em = prediction_mask
    acc, prec, rec, f1 = sergio_metrics(gt_grn, prediction_mask, false_negative_base)
    return acc

gt = sys.argv[4]
if gt == 'True':
    groundtruth = True
else:
    groundtruth = False

def exp_acc(gt_exp, betaem):
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
    lr = trial.suggest_float('lrs', 1e-6, 0.1)
    alpha = trial.suggest_float('a', 0.1, 1)
    beta = trial.suggest_float('b', 0.1, 1)
    explainer = NodeBetaExplainer.BetaExplainer(model, x, edge_index, torch.device('cpu'), alpha, beta)
    explainer.train(5, lr)
    betaem = explainer.edge_mask()
    faith = faithfulness(model, x, edge_index, betaem)
    res = 1 - faith
    if groundtruth:
        accuracy, prec, rec, f1 = exp_acc(gt_exp, betaem)
        res += accuracy
        res /= 2
    return res

def graph_objective(trial):
    lr = trial.suggest_float('lrs', 1e-6, 0.1)
    alpha = trial.suggest_float('a', 0.1, 1)
    beta = trial.suggest_float('b', 0.1, 1)
    explainer = GraphBetaExplainer.BetaExplainer(model, graph_data, edge_index, torch.device('cpu'), num_graphs, alpha, beta)
    explainer.train(5, lr)
    betaem = explainer.edge_mask()
    faith = graph_faithfulness(model, graph_data, edge_index, betaem)
    res = 1 - faith
    if groundtruth:
        accuracy, prec, rec, f1 = exp_acc(gt_exp, betaem)
        res += accuracy
        res /= 2
    return res
    
pruner = optuna.pruners.MedianPruner()
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), pruner=pruner)
if sys.argv[1] in shapeggen:
    study.optimize(shapeggen_objective, n_trials=150)
elif sys.argv[1] in sergio:
    study.optimize(sergio_objective, n_trials=150)
else:
    if sys.argv[2] == 'node':
        study.optimize(node_objective, n_trials=150)
    else:
        study.optimize(graph_objective, n_trials=150)
print('Best hyperparameters:', study.best_params)
print('Best Result:', study.best_value)

lr = study.best_params['lrs']
alpha = study.best_params['a']
beta = study.best_params['b']
ep = 500
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
            best_faith = faithfulness(model, x, edge_index, exp)
    print(f'Best Accuracy: {best_acc}, Best Precision: {best_prec}, Best Recall: {best_rec}, Best F1 Score: {best_f1}., Best Unfaithfulness: {best_faith}')
elif sys.argv[1] in sergio:
    explainer = GraphBetaExplainer.BetaExplainer(model, graph_data, edge_index, torch.device('cpu'), 2000, alpha, beta)
    explainer.train(ep, lr)
    prediction_mask = explainer.edge_mask()
    em = prediction_mask
    acc, prec, rec, f1 = sergio_metrics(gt_grn, prediction_mask, false_negative_base)
    faith = graph_faithfulness(model, graph_data, edge_index, edge_mask)
    print(f'Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1 Score: {f1}, Unfaithfulness: {faith}')
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
    else:
        print(f'Unfaithfulness: {faith}')