from ShapeGGenSimulator import ShapeGGen, return_ShapeGGen_dataset, graph_exp_acc, set_seed, SubgraphX
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
from ShapeGGenSimulator import GNNExplainer as ShapeGGenGNNExplainer
# Takes 1 argument
# Argument 1: Denotes data type; specify 25 or 50 for the 25% and 50% sparse SERGIO datasets; base, hetero, unfair, lessinform, or moreinform for ShapeGGen Simulator datasets
def get_sergio_data(num):
    '''This function allows us to load our data. We use the subgraph data - the original graph plus
        extra points associated with different points that are highly correlated to see if our explainers
        capture the ground truth data well'''
    labels = np.load(f'Time Experiments/sergio data/SERGIOsimu_{num}Sparse_noLibEff_cTypes.npy') # get labels
    features = np.load(f'Time Experiments/sergio data/SERGIOsimu_{num}Sparse_noLibEff_concatShuffled.npy') # get node features
    num_features = features.shape[1]
    num_classes = len(np.unique(labels))
    adj = np.load(f'Time Experiments/sergio data/ExtraPointsSergio{num}.npy') # get subgraph based on correlation --> contains false edges, some true edges, misses some true edges
    gt = np.load('Time Experiments/sergio data/gt_adj.npy') # adjacency of all true edges
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
    # the false_negative_base is the number of elements in the groundtruth absent from the adjacency used for the model and explainer
    # explainers will not be able to indicate the importance of these edges, so they are inherently false negatives
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
    gt_grn = gt_grn.numpy()
    prediction_mask = prediction_mask.numpy()
    for ct in range(0, gt_grn.shape[0]):
        if gt_grn[ct] == 1:
            if prediction_mask[ct] > 0.5:
                tp += 1
            else:
                fn += 1
        else:
            if prediction_mask[ct] <= 0.5:
                tn += 1
            else:
                fp += 1
    acc = (tp + tn) / (tp + tn + fn + fp)
    if tp + fp == 0:
        prec = 0
    else:
        prec = tp / (tp + fp)
    if tp + fp == 0:
        rec = 0
    else:
        rec = tp / (tp + fn)
    if prec != 0 and rec != 0:
        f1 = (2 * prec * rec) / (prec + rec)
    else:
        f1 = 0
    return acc, prec, rec, f1
shapeggen = ['base', 'heterophilic', 'unfair', 'moreinform', 'lessinform', 'test']
sergio = ['25', '50']
if sys.argv[1] in shapeggen:
    ''' Set up ShapeGGen dataset and various associated parameters for the model/explanation such as the number of features.'''
    if sys.argv[1] == 'lessinform':
        num_features = 22
    else:
        num_features = 12
    x, y, edge_index, gt_exp, train_mask, test_mask, val_mask = return_ShapeGGen_dataset(sys.argv[1])
    num_classes = np.unique(y.numpy()).shape[0]
    num_hops = 3
elif sys.argv[1] in sergio:
    ''' Set up the SERGIO dataset if chosen'''
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
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
else:
    '''Set up the Texas dataset'''
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
''' Get the number of features for the model '''
if sys.argv[1] in 'sergio':
    input_features = 1
else:
    input_features = num_features
device = torch.device('cpu')
# Import files for running BetaExplainers
import NodeBetaExplainer, GraphBetaExplainer
criterion = torch.nn.CrossEntropyLoss()
class SERGIOGCN(torch.nn.Module):
    def __init__(self, output_size):
        ''' Define the model chosen for the SERGIO datasets '''
        super(SERGIOGCN, self).__init__()
        self.conv1 = SAGEConv(1, output_size)
        self.embedding_size = output_size
    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None: # No batch given
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            batch = torch.zeros(x.size(0), dtype=torch.long).to(device)
        x = self.conv1(x, edge_index, edge_weights)
        x = F.dropout(x, p=0.2, training=self.training)
        x = global_max_pool(x, batch)
        return x
class xAIGCN(torch.nn.Module):
    def __init__(self, hidden_channels, input_feat, classes):
        ''' Define the model for the ShapeGGen datasets and the Texas dataset as node classification problems '''
        super(xAIGCN, self).__init__()
        self.conv1 = GCNConv(input_feat, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, classes)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
y_true = y.numpy()
# Get GNN training parameters for the given datasets (number of epochs, learning rate, and weight decay) and define the GNN used
if sys.argv[1] in shapeggen:
    num_epochs = 2000
    if sys.argv[1] == 'base':
        lr = 0.16
        wd = 0.0001
    elif sys.argv[1] == 'hetero':
        lr = 0.1
        wd = 5e-5
    elif sys.argv[1] == 'unfair':
        lr = 0.15
        wd = 0.0001
    elif sys.argv[1] == 'moreinform':
        lr = 0.05
        wd = 0.001
    elif sys.argv[1] == 'lessinform':
        lr = 0.05
        wd = 0.001
    else:
        lr = 0.05
        wd = 0.001
    model = xAIGCN(16, input_features, num_classes)
elif sys.argv[1] in sergio:
    set_seed(200)
    num_epochs = 50
    lr = 0.001
    wd = 0
    model = SERGIOGCN(num_classes)
else:
    lr = 0.03076
    wd = 0.2682
    num_epochs = 250
    set_seed(0)
    model = xAIGCN(1703, input_features, num_classes)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
# Train the model based n the classification type
if sys.argv[1] in shapeggen or sys.argv[1] == 'Texas':
    # Train the model if it's a node classification dataset (Texas, and ShapeGGen datasets)
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
    # Train model if it's a graph classification dataset (SERGIO datasets)
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
# Define parameters for datasets for all explanation types
if sys.argv[1] in sergio:
    if sys.argv[1] == '25':
        lr = 0.001
        alpha = 0.55
        beta = 0.65
        gnnlr = 0.00001
        gnnty = 'phenomenon'
        gnnbat = 300
        rollout = 20
        min_atoms = 1
        c_puct = 8.165129763712843
        expand_atoms = 5
        sample_num = 6
    else:
        lr = 0.01
        alpha = 0.05
        beta = 0.95
        gnnlr = 0.0001
        gnnty = 'phenomenon'
        gnnbat = 859
        rollout = 4
        min_atoms = 1
        c_puct = 2.0727783316948987
        expand_atoms = 7
        sample_num = 9
elif sys.argv[1] in shapeggen:
    if sys.argv[1] == 'base':
        lr = 0.05
        alpha = 0.8
        beta = 0.6
        gnnlr = 1e-5
        rollout = 5
        min_atoms = 9
        c_puct = 4.763396105210084
        expand_atoms = 16
        sample_num = 10
        subgraphxidx = 8966
    elif sys.argv[1] == 'hetero':
        lr = 0.05
        alpha = 0.7
        beta = 0.6
        gnnlr = 1e-5
        rollout = 6
        min_atoms = 10
        c_puct = 7.948316238894563
        expand_atoms = 6
        sample_num = 3
        subgraphxidx = 7324
    elif sys.argv[1] == 'unfair':
        lr = 0.05
        alpha = 0.8
        beta = 0.6
        gnnlr = 1e-5
        rollout = 8
        min_atoms = 7
        c_puct = 6.933836891280444
        expand_atoms = 8
        sample_num = 8
        subgraphxidx = 3046
    elif sys.argv[1] == 'lessinform':
        lr = 0.05
        alpha = 0.6
        beta = 0.6
        gnnlr = 1e-5
        rollout = 5
        min_atoms = 4
        c_puct = 4.919019371456805
        expand_atoms = 4
        sample_num = 3
        subgraphxidx = 2913
    elif sys.argv[1] == 'moreinform':
        lr = 0.05
        alpha = 0.8
        beta = 0.8
        gnnlr = 1e-5
        rollout = 7
        min_atoms = 9
        c_puct = 7.318672180500459
        expand_atoms = 3
        sample_num = 10
        subgraphxidx = 3403
    else:
        lr = 0.05
        alpha = 0.8
        beta = 0.6
        gnnlr = 1e-5
        rollout = 7
        min_atoms = 9
        c_puct = 7.318672180500459
        expand_atoms = 3
        sample_num = 10
        subgraphxidx = 5
else:
    lr = 0.02649
    alpha = 1.76
    beta = 1.866
    gnnlr = 0.29989
    gnnty = 'model'
    rollout = 16
    min_atoms = 4
    c_puct = 5.89811 
    expand_atoms = 5
    sample_num = 8
    subgraphxidx = 53
ep = 25
# define GNNExplainer training datasets
if sys.argv[1] in shapeggen:
    gnnep = 200
else:
    gnnep = 300
results = []
graphs = []
for run in range(0, 10):
    # Run explainer datasets across 10 random scenes to get a sense of variability of explainers
    # add results and record probabilities for each edge for all graphs
    seed = np.random.randint(0, 1000001)
    set_seed(int(seed))
    if sys.argv[1] in shapeggen:
        # Get BetaExplainer results
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
                best_faith = faithfulness(model, x, ei exp)
                best_exp = exp.numpy()
                best_ei = ei.numpy()
                best_gt = gt_exp[i].edge_imp.numpy()
        print(f'BetaExplainer Best Accuracy: {best_acc}, Best Precision: {best_prec}, Best Recall: {best_rec}, Best F1 Score: {best_f1}., Best Unfaithfulness: {best_faith}')
        out = [seed, best_acc, best_prec, best_rec, best_f1, best_faith, 'BetaExplainer']
        results.append(out)
        for i in range(0, best_exp.shape[0]):
            graphs.append([seed, best_exp[i], best_ei[0, i], best_ei[1, i], best_gt[i], 'BetaExplainer'])
        # Get GNNExplainer results
        explainer = ShapeGGenGNNExplainer(model)
        expgnn = explainer.get_explanation_graph(x, edge_index, lr = gnnlr, ep=ep)
        betaem = expgnn.edge_imp
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
        print(f'GNNExplainer Best Accuracy: {best_acc}, Best Precision: {best_prec}, Best Recall: {best_rec}, Best F1 Score: {best_f1}., Best Unfaithfulness: {best_faith}')
        out = [seed, best_acc, best_prec, best_rec, best_f1, best_faith, 'GNNExplainer']
        results.append(out)
        for i in range(0, best_exp.shape[0]):
            graphs.append([seed, best_exp[i], best_ei[0, i], best_ei[1, i], best_gt[i], 'GNNExplainer'])
        # Get SubgraphX results
        explainer = SubgraphX(model, rollout = rollout, min_atoms = min_atoms, c_puct = c_puct, expand_atoms = expand_atoms, sample_num = sample_num)
        expgnn = explainer.get_explanation_node(x, edge_index, subgraphxidx, y = y)
        betaem = expgnn.edge_imp
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
        print(f'SubgraphX Best Accuracy: {best_acc}, Best Precision: {best_prec}, Best Recall: {best_rec}, Best F1 Score: {best_f1}., Best Unfaithfulness: {best_faith}')
        out = [seed, best_acc, best_prec, best_rec, best_f1, best_faith, 'SubgraphX']
        results.append(out)
        for i in range(0, best_exp.shape[0]):
            graphs.append([seed, best_exp[i], best_ei[0, i], best_ei[1, i], best_gt[i], 'SubgraphX'])
    elif sys.argv[1] in sergio:
        # Get BetaExplainer results
        explainer = GraphBetaExplainer.BetaExplainer(model, graph_data, edge_index, torch.device('cpu'), 2000, alpha, beta)
        explainer.train(ep, lr)
        prediction_mask = explainer.edge_mask()
        em = prediction_mask
        acc, prec, rec, f1 = sergio_metrics(gt_grn, prediction_mask, false_negative_base)
        faith = graph_faithfulness(model, graph_data, edge_index, em)
        print(f'BetaExplainer Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1 Score: {f1}, Unfaithfulness: {faith}')
        out = [seed, acc, prec, rec, f1, faith, 'BetaExplainer']
        results.append(out)
        em = em.numpy()
        ei = edge_index.numpy()
        gt = gt_grn.numpy()
        for i in range(0, em.shape[0]):
            graphs.append([seed, em[i], ei[0, i], ei[1, i], gt[i], 'BetaExplainer'])
        # Get GNNExplainer results
        expgnn = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=gnnep, lr=gnnlr),
            explanation_type=gnnty,
            edge_mask_type='object',
            node_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='graph',
                return_type='log_probs',
            ),
        )
        loader = graph_data
        bat = 0
        for batch in loader:
            if bat < gnnbat:
                bat += 1
            elif bat == gnnbat:
                explana = expgnn(torch.reshape(batch.x, (batch.x.shape[0], 1)), batch.edge_index, target=batch.y)
                prediction_mask = explana.edge_mask
                bat += 1
            else:
                break
        em = prediction_mask
        acc, prec, rec, f1 = sergio_metrics(gt_grn, prediction_mask, false_negative_base)
        faith = graph_faithfulness(model, graph_data, edge_index, em)
        print(f'GNNExplainer Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1 Score: {f1}, Unfaithfulness: {faith}')
        out = [seed, acc, prec, rec, f1, faith, 'GNNExplainer']
        results.append(out)
        em = em.numpy()
        ei = edge_index.numpy()
        gt = gt_grn.numpy()
        for i in range(0, em.shape[0]):
            graphs.append([seed, em[i], ei[0, i], ei[1, i], gt[i], 'GNNExplainer'])
        # Get SubgraphX results
        explainer = SubgraphX(model, rollout = rollout, min_atoms = min_atoms, c_puct = c_puct, expand_atoms = expand_atoms, sample_num = sample_num)
        subgraphxbat = 0
        bat = 0
        for batch in loader:
            if bat == subgraphxbat:
                expgnn = explainer.get_explanation_graph(torch.reshape(batch.x, (batch.x.shape[0], 1)), edge_index)
            else:
                break
        prediction_mask = expgnn.edge_imp
        em = prediction_mask
        acc, prec, rec, f1 = sergio_metrics(gt_grn, prediction_mask, false_negative_base)
        faith = graph_faithfulness(model, graph_data, edge_index, em)
        print(f'SubgraphX Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1 Score: {f1}, Unfaithfulness: {faith}')
        out = [seed, acc, prec, rec, f1, faith, 'SubgraphX']
        results.append(out)
        em = em.numpy()
        ei = edge_index.numpy()
        gt = gt_grn.numpy()
        for i in range(0, em.shape[0]):
            graphs.append([seed, em[i], ei[0, i], ei[1, i], gt[i], 'SubgraphX'])
    else:
        # Get BetaExplainer results
        explainer = NodeBetaExplainer.BetaExplainer(model, x, edge_index, torch.device('cpu'), alpha, beta)
        explainer.train(ep, lr)
        betaem = explainer.edge_mask()
        faith = faithfulness(model, x, edge_index, betaem)
        em = betaem.numpy()
        sparse = 0
        for i in range(0, em.shape[0]):
            if em[i] >= 0.5:
                sparse += 1
        sparse /= em.shape[0]
        out = [seed, faith, sparse, 'BetaExplainer']
        results.append(out)
        print(f'BetaExplainer Faithfulness: {faith}, Fraction of Kept Edges: {sparse}')
        ei = edge_index.numpy()
        for i in range(0, em.shape[0]):
            graphs.append([seed, em[i], ei[0, i], ei[1, i], 'BetaExplainer'])
        # Get GNNExplainer results
        expgnn = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=gnnep, lr=gnnlr),
            explanation_type=gnnty,
            edge_mask_type='object',
            node_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='log_probs',
            ),
        )
        explana = expgnn(x, edge_index, target=y)
        betaem = explana.edge_mask
        em = betaem.numpy()
        sparse = 0
        for i in range(0, em.shape[0]):
            if em[i] >= 0.5:
                sparse += 1
        sparse /= em.shape[0]
        faith = faithfulness(model, x, edge_index, betaem)
        out = [seed, faith, sparse, 'GNNExplainer']
        results.append(out)
        print(f'GNNExplainer Faithfulness: {faith}, Fraction of Kept Edges: {sparse}')
        ei = edge_index.numpy()
        for i in range(0, em.shape[0]):
            graphs.append([seed, em[i], ei[0, i], ei[1, i], 'GNNExplainer'])
        # Get SubgraphX results
        explainer = SubgraphX(model, rollout = rollout, min_atoms = min_atoms, c_puct = c_puct, expand_atoms = expand_atoms, sample_num = sample_num)
        expgnn = explainer.get_explanation_node(x, edge_index, subgraphxidx, y = y)
        betaem = expgnn.edge_imp
        em = betaem.numpy()
        sparse = 0
        for i in range(0, em.shape[0]):
            if em[i] >= 0.5:
                sparse += 1
        sparse /= em.shape[0]
        faith = faithfulness(model, x, edge_index, betaem)
        out = [seed, faith, sparse, 'SubgraphX']
        results.append(out)
        print(f'SubgraphX Faithfulness: {faith}, Fraction of Kept Edges: {sparse}')
        ei = edge_index.numpy()
        for i in range(0, em.shape[0]):
            graphs.append([seed, em[i], ei[0, i], ei[1, i], 'SubgraphX'])
if len(out) > 4:
    cols = ['Seed', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Unfaithfulness', 'Explainer']
    cols1 = ['Seed', 'Probability', 'P1', 'P2', 'Groundtruth', 'Explainer']
else:
    cols = ['Seed', 'Unfaithfulness', 'Kept Edges', 'Explainer']
    cols1 = ['Seed', 'Probability', 'P1', 'P2', 'Explainer']
df = pd.DataFrame(results, columns=cols)
fn = sys.argv[1]
df.to_csv(f'SeedResults{fn}.csv')
df1 = pd.DataFrame(graphs, columns=cols1)
fn = sys.argv[1]
df1.to_csv(f'SeedGraphResults{fn}.csv')
results_df = df.copy()
all_graphs = df1.copy()
# Return average metrics for datasets taking into account the Texas dataset does not have a notion of groundtruth --> only faithfulness and number of edge indices kept
# Return the seed for the best performing metrics for each explainer to choose this seed as to chose this explanation to graph
if 'Accuracy' in cols:
    df = results_df[results_df['Explainer'] == 'BetaExplainer']
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
    print(f'BetaExplainer Average Accuracy: {acc}, Average Precision: {prec}, Average Recall: {rec}, Average F1 Score: {f1}, Average Unfaithfulness: {unfaith}')
    best_acc = 0
    best_f1 = 0
    best_faith = 1
    betaidx = 0
    for i in range(0, len(a)):
        if a[i] >= best_acc and fs[i] >= best_f1 and un[i] <= best_faith:
            betaidx = i
            best_acc = a[i]
            best_f1 = fs[i]
            best_faith = un[i]
    df = results_df[results_df['Explainer'] == 'GNNExplainer']
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
    print(f'GNNExplainer Average Accuracy: {acc}, Average Precision: {prec}, Average Recall: {rec}, Average F1 Score: {f1}, Average Unfaithfulness: {unfaith}')
    best_acc = 0
    best_f1 = 0
    best_faith = 1
    gnnidx = 0
    for i in range(0, len(a)):
        if a[i] >= best_acc and fs[i] >= best_f1 and un[i] <= best_faith:
            gnnidx = i
            best_acc = a[i]
            best_f1 = fs[i]
            best_faith = un[i]
    df = results_df[results_df['Explainer'] == 'SubgraphX']
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
    print(f'SubgraphX Average Accuracy: {acc}, Average Precision: {prec}, Average Recall: {rec}, Average F1 Score: {f1}, Average Unfaithfulness: {unfaith}')
    best_acc = 0
    best_f1 = 0
    best_faith = 1
    subgraphxidx = 0
    for i in range(0, len(a)):
        if a[i] >= best_acc and fs[i] >= best_f1 and un[i] <= best_faith:
            subgraphxidx = i
            best_acc = a[i]
            best_f1 = fs[i]
            best_faith = un[i]
else:
    df = results_df[results_df['Explainer'] == 'BetaExplainer']
    f = list(df['Unfaithfulness'])
    unfaith = np.mean(f)
    k = list(df['Kept Edges'])
    sparse = np.mean(k)
    print(f'BetaExplainer Average Unfaithfulness: {unfaith}, Average Fraction of Kept Edges: {sparse}')
    best_faith = 1
    best_sparse = 1
    betaidx = 0
    for i in range(0, len(f)):
        if f[i] <= best_faith and k[i] <= best_sparse:
            betaidx = i
            best_faith = f[i]
            best_sparse = k[i]
    df = results_df[results_df['Explainer'] == 'GNNExplainer']
    f = list(df['Unfaithfulness'])
    unfaith = np.mean(f)
    k = list(df['Kept Edges'])
    sparse = np.mean(k)
    print(f'GNNExplainer Average Unfaithfulness: {unfaith}, Average Fraction of Kept Edges: {sparse}')
    best_faith = 1
    best_sparse = 1
    gnnidx = 0
    for i in range(0, len(f)):
        if f[i] <= best_faith and k[i] <= best_sparse:
            gnnidx = i
            best_faith = f[i]
            best_sparse = k[i]
    df = results_df[results_df['Explainer'] == 'SubgraphX']
    f = list(df['Unfaithfulness'])
    unfaith = np.mean(f)
    k = list(df['Kept Edges'])
    sparse = np.mean(k)
    print(f'SubgraphX Average Unfaithfulness: {unfaith}, Average Fraction of Kept Edges: {sparse}')
    best_faith = 1
    best_sparse = 1
    subgraphxidx = 0
    for i in range(0, len(f)):
        if f[i] <= best_faith and k[i] <= best_sparse:
            subgraphxidx = i
            best_faith = f[i]
            best_sparse = k[i]
subsetbeta = results_df[results_df['Explainer'] == 'BetaExplainer']
subsetgnn = results_df[results_df['Explainer'] == 'GNNExplainer']
subsetsubgraph = results_df[results_df['Explainer'] == 'SubgraphX']
best_beta_seed = list(subsetbeta['Seed'])[betaidx]
best_gnn_seed = list(subsetgnn['Seed'])[gnnidx]
best_subgraphx_seed = list(subsetsubgraph['Seed'])[subgraphxidx]
print(best_beta_seed, best_gnn_seed, best_subgraphx_seed)
if sys.argv[1] in shapeggen or sys.argv[1] == 'Texas':
    num_nodes = x.shape[0]
else:
    num_nodes = num_features
df1 = all_graphs.copy()
# graph best BetaExplanation explanation graph
df1 = df1[df1['Seed'] == best_beta_seed]
df1 = df1[df1['Explainer'] == 'BetaExplainer']
b1 = list(df1['P1'])
b2 = list(df1['P2'])
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
if sys.argv[1] != 'Texas':
    # non-Texas graphs have a notion of groundtruth
    # we make a set of the groundtruth edges tp_set to check edges against
    tp_edges = df1[df1['Groundtruth'] == 1]
    tp_set = set()
    p1s = list(tp_edges['P1'])
    p2s = list(tp_edges['P2'])
    for i in range(len(p1s)):
        p1 = p1s[i]
        p2 = p2s[i]
        tp_set.add((p1, p2))
    true_edge = '#1A85FF' # denote color of true positive edges (light blue)
    false_positive_edge = '#D41159' # denote color of false positive edges (magenta)
    false_negative_edge = '#ED9FBC' # denote color of false negative edges (light pink)
for i in range(0, len(b1)):
    p1 = b1[i]
    p2 = b2[i]
    if probs[i] >= 0.5:
        # add chosen edges --> those with probability of at least 0.5
        if sys.argv[1] == 'Texas':
            G.add_edge(p1, p2) # no edge color added for Texas as there isn't a notion of groundtruth
        else:
            # for datasets with groundtruth, denote associated color of edge
            if (p1, p2) in tp_set:
                color = true_edge # if it's an edge in the groundtruth, denote it as a blue color
            else:
                color = false_positive_edge # if it's an edge not in the groundtruth, denote it as a magenta edge to indicate it's inaccurate
            G.add_edge(p1, p2, color=color)
        p = (probs[i] - mn + 1e-5) / (mx - mn) # add edge weighting based on edge probability
        #p = probs[i]
        weights.append(5 * p)
    else:
        # for datasets with groundtruth, add edges that are false negatives, labeling them with light pink
        # this will indicate that the edge should be in the explanation but isn't
        if sys.argv[1] != 'Texas' and (p1, p2) in tp_set:
            G.add_edge(p1, p2, color=false_negative_edge)
h = ig.Graph.from_networkx(G)
ig.plot(h, vertex_size=7, edge_width=weights, target=f'{sys.argv[1]}BetaExplainerPlot.png')
# graph best GNNExplanation explanation graph
df1 = all_graphs.copy()
df1 = df1[df1['Seed'] == best_gnn_seed]
df1 = df1[df1['Explainer'] == 'GNNExplainer']
b1 = list(df1['P1'])
b2 = list(df1['P2'])
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
        col = color[actual[node]]
        G.add_node(node, color=col)
lst = []
weights = []
probs = list(df1['Probability'])
mx = np.max(probs)
mn = np.min(probs)
if sys.argv[1] != 'Texas':
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
for i in range(0, len(b1)):
    p1 = b1[i]
    p2 = b2[i]
    if probs[i] >= 0.5:
        if sys.argv[1] == 'Texas':
            G.add_edge(p1, p2)
        else:
            if (p1, p2) in tp_set:
                color = true_edge
            else:
                color = false_positive_edge
            G.add_edge(p1, p2, color=color)
        p = (probs[i] - mn + 1e-5) / (mx - mn)
        #p = probs[i]
        weights.append(5 * p)
    else:
        if sys.argv[1] != 'Texas' and (p1, p2) in tp_set:
            G.add_edge(p1, p2, color=false_negative_edge)
h = ig.Graph.from_networkx(G)
ig.plot(h, vertex_size=7, edge_width=weights, target=f'{sys.argv[1]}GNNExplainerPlot.png')
# graph best SubgraphX explanation graph
df1 = all_graphs.copy()
df1 = df1[df1['Seed'] == best_subgraphx_seed]
df1 = df1[df1['Explainer'] == 'SubgraphX']
b1 = list(df1['P1'])
b2 = list(df1['P2'])
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
        col = color[actual[node]]
        G.add_node(node, color=col)
lst = []
weights = []
probs = list(df1['Probability'])
mx = np.max(probs)
mn = np.min(probs)
if sys.argv[1] != 'Texas':
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
for i in range(0, len(b1)):
    p1 = b1[i]
    p2 = b2[i]
    if probs[i] >= 0.5:
        if sys.argv[1] == 'Texas':
            G.add_edge(p1, p2)
        else:
            if (p1, p2) in tp_set:
                color = true_edge
            else:
                color = false_positive_edge
            G.add_edge(p1, p2, color=color)
        p = (probs[i] - mn + 1e-5) / (mx - mn)
        #p = probs[i]
        weights.append(5 * p)
    else:
        if sys.argv[1] != 'Texas' and (p1, p2) in tp_set:
            G.add_edge(p1, p2, color=false_negative_edge)
h = ig.Graph.from_networkx(G)
ig.plot(h, vertex_size=7, edge_width=weights, target=f'{sys.argv[1]}SubgraphXPlot.png')
