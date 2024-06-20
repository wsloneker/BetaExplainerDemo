import sys
import os.path as osp
import os
import numpy as np
from numpy.random import RandomState
import random
import copy
import multiprocessing
import matplotlib.pyplot as plt
import networkx as nx
import time
import matplotlib as mpl
import torch
import torch_geometric
from torch_geometric.data import Data, DataLoader
import torch_geometric.transforms as T
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
from torch_geometric.nn import GCNConv, SAGEConv, ChebConv, GatedGraphConv, Linear, global_mean_pool, global_max_pool
from torch.nn import functional as F
import pickle as pkl
import pandas as pd
import csv
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import sklearn.metrics

def set_seed(seed: int = 42) -> None:
    '''This function allows us to set the seed for the notebook across different seeds.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
    
set_seed(200)

def get_data(num):
    '''This function allows us to load our data. We use the supergraph data - the original graph plus
        extra points associated with differnet points that are highly correlated to see if our explainers
        capture the ground truth data well'''
    labels = np.load(f'sergio data/SERGIOsimu_{num}Sparse_noLibEff_cTypes.npy')
    features = np.load(f'sergio data/SERGIOsimu_{num}Sparse_noLibEff_concatShuffled.npy')
    num_features = features.shape[1]
    num_classes = len(np.unique(labels))
    adj = np.load(f'sergio data/ExtraPointsSergio{num}.npy')
    return adj, features, labels, num_features, num_classes
    
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, data, output_size):
        super(GCN, self).__init__()
        self.conv1 = SAGEConv(1, hidden_channels)
        self.embedding_size = hidden_channels
    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None: # No batch given
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            batch = torch.zeros(x.size(0), dtype=torch.long).to(device)
        x = self.conv1(x, edge_index, edge_weights)
        x = F.dropout(x, p=0.2, training=self.training)
        x = global_max_pool(x, batch)
        return x

def train(model, train_loader):
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
    return avgLoss / 47

def test(model, loader):
    model.eval()
    correct = 0
    avgAUC = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data.x = torch.reshape(data.x, (data.x.shape[0], 1))
        data.x = data.x.type(torch.FloatTensor)
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        # one_hot = np.eye(2)[data.y.cpu()]
        # results = (out.cpu().detach().numpy() == np.max(out.cpu().detach().numpy(), axis=1, keepdims=True)).view('i1')  
        # roc = roc_auc_score(one_hot, results)
        # avgAUC += roc
    return correct / len(loader.dataset), avgAUC / len(loader)  # Derive ratio of correct predictions.

import torch.nn.functional as F
import torch.distributions.constraints as constraints
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO

class DNFGExplainer:
    def __init__(self, model: torch.nn.Module, splines: int, X: torch.Tensor, G: torch.Tensor, device: torch.device, sz: int):
        self.model = model
        self.n_splines = splines
        self.X = X
        self.G = G
        self.device = device
        self.sz = sz
        with torch.no_grad():
            self.target = torch.zeros(self.sz, 2)
            i = 0
            for data in self.X:  # Iterate in batches over the training/test dataset.
                data.x = torch.reshape(data.x, (data.x.shape[0], 1))
                data.x = data.x.type(torch.FloatTensor)
                data = data.to(self.device)
                out = self.model(data.x, self.G, data.batch)
                self.target[i, 0] = out[0, 0]
                self.target[i, 1] = out[0, 1]
                i += 1
            self.target = self.target.flatten()
        self.ne = G.shape[1]

        self.base_dist = dist.MultivariateNormal(torch.zeros(self.ne).to(self.device), torch.eye(self.ne).to(self.device))
        self.splines = []
        self.params_l = []
        for _ in range(self.n_splines):
            self.splines.append(T.spline_autoregressive(self.ne).to(self.device))
            self.params_l += self.splines[-1].parameters()
        self.params = torch.nn.ParameterList(self.params_l)
        self.flow_dist = dist.TransformedDistribution(self.base_dist, self.splines)

    def forward(self):
        m = self.flow_dist.rsample(torch.Size([25,])).sigmoid().mean(dim=0)
        set_masks(self.model, m, self.G, False)
        preds = torch.zeros(self.sz, 2)
        i = 0
        for data in self.X:  # Iterate in batches over the training/test dataset.
            data.x = torch.reshape(data.x, (data.x.shape[0], 1))
            data.x = data.x.type(torch.FloatTensor)
            data = data.to(self.device)
            out = self.model(data.x, self.G, data.batch)
            preds[i, 0] = out[0, 0]
            preds[i, 1] = out[0, 1]
            i += 1
        preds = preds.flatten()
        return preds, m

    def edge_mask(self):
        return self.flow_dist.sample(torch.Size([10000, ])).sigmoid().mean(dim=0)

    def edge_distribution(self):
        return self.flow_dist.sample(torch.Size([10000, ])).sigmoid()

    def train(self, epochs: int, lr: float):
        optimizer = torch.optim.Adam(self.params, lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            preds, m = self.forward()
            kl = F.kl_div(preds, self.target, log_target=True)
            reg = m.mean()
            loss = kl + 1e-5 * reg
            # if (epoch + 1) % 250 == 0:
            #     print(f"epoch = {epoch + 1} | loss = {loss.detach().item()}")
            loss.backward()
            optimizer.step()
            self.flow_dist.clear_cache()
        clear_masks(self.model)

    def clean(self):
        cpu = torch.device('cpu')
        for spl in self.splines:
            spl = spl.to(cpu)
        for p in self.params_l:
            p = p.to(cpu)
        self.params = self.params.to(cpu)
        self.X = self.X.to(cpu)
        self.G = self.G.to(cpu)

        del self.base_dist
        del self.splines
        del self.params_l
        del self.params
        del self.flow_dist
        del self.X
        del self.G

class BetaExplainer:
    '''This is a post-hoc explainer based on the Beta Distribution.'''
    def __init__(self, model: torch.nn.Module, data: torch.Tensor, G: torch.Tensor, device: torch.device, sz, a=0.7, b=0.9):
        '''Initialization of the model.'''
        self.model = model
        self.X = data
        self.G = G
        self.sz = sz
        with torch.no_grad():
            self.target = torch.zeros(self.sz, 2)
            i = 0
            for data in self.X:  # Iterate in batches over the training/test dataset.
                data.x = torch.reshape(data.x, (data.x.shape[0], 1))
                data.x = data.x.type(torch.FloatTensor)
                data = data.to(device)
                out = self.model(data.x, self.G, data.batch)
                self.target[i, 0] = out[0, 0]
                self.target[i, 1] = out[0, 1]
                i += 1
            self.target = self.target.flatten()

        self.ne = G.shape[1]
        self.N = self.sz
        self.obs = 1000
        self.device = device
        self.a = a
        self.b = b

    def model_p(self, ys):
        alpha = self.a * torch.ones(self.N).to(self.device)
        beta = self.b * torch.ones(self.N).to(self.device)
        alpha_edges = alpha[self.G[0, :]]
        beta_edges = beta[self.G[1, :]]
        m = pyro.sample("mask", dist.Beta(alpha_edges, beta_edges).to_event(1))
        set_masks(self.model, m, self.G, False)
        preds = torch.zeros(self.sz, 2)
        i = 0
        for data in self.X:  # Iterate in batches over the training/test dataset.
            data.x = torch.reshape(data.x, (data.x.shape[0], 1))
            data.x = data.x.type(torch.FloatTensor)
            data = data.to(self.device)
            out = self.model(data.x, self.G, data.batch)
            preds[i, 0] = out[0, 0]
            preds[i, 1] = out[0, 1]
            i += 1
        preds = preds.exp().flatten()
        with pyro.plate("data_loop"):
            pyro.sample("obs", dist.Categorical(preds), obs=ys)

    def guide(self, ys):
        alpha = pyro.param("alpha_q", self.a * torch.ones(self.N).to(self.device), constraint=constraints.positive)
        beta = pyro.param("beta_q", self.b * torch.ones(self.N).to(self.device), constraint=constraints.positive)
        alpha_edges = alpha[self.G[0, :]]
        beta_edges = beta[self.G[1, :]]
        m = pyro.sample("mask", dist.Beta(alpha_edges, beta_edges).to_event(1))
        set_masks(self.model, m, self.G, False)
        init = torch.zeros(self.sz, 2)
        i = 0
        for data in self.X:  # Iterate in batches over the training/test dataset.
            data.x = torch.reshape(data.x, (data.x.shape[0], 1))
            data.x = data.x.type(torch.FloatTensor)
            data = data.to(self.device)
            out = self.model(data.x, self.G, data.batch)
            init[i, 0] = out[0, 0]
            init[i, 1] = out[0, 1]
            i += 1
        init.exp().flatten()

    def train(self, epochs: int, lr: float = 0.0005):
        adam_params = {"lr": lr, "betas": (0.90, 0.999)}
        optimizer = Adam(adam_params)
        svi = SVI(self.model_p, self.guide, optimizer, loss=Trace_ELBO())

        elbos = []
        for epoch in range(epochs):
            ys = torch.distributions.categorical.Categorical(self.target.exp()).sample(torch.Size([self.obs]))
            elbo = svi.step(ys)
            elbos.append(elbo)
            if epoch > 249:
                elbos.pop(0)

        clear_masks(self.model)

    def edge_mask(self):
        m = torch.distributions.beta.Beta(pyro.param("alpha_q").detach()[self.G[0, :]], pyro.param("beta_q").detach()[self.G[1, :]]).sample(torch.Size([10000]))
        return m.mean(dim=0)

    def edge_distribution(self):
        return torch.distributions.beta.Beta(pyro.param("alpha_q").detach()[self.G[0, :]], pyro.param("beta_q").detach()[self.G[1, :]]).sample(torch.Size([10000]))
        
def get_explain_and_comp_edges(full_adj_matrix, edge_mask):
    lst1 = full_adj_matrix[0]
    lst2 = full_adj_matrix[1]
    l1 = []
    l2 = []
    l3 = []
    l4 = []
    for i in range(0, len(lst1)):
        if em[i] > 0.5:
            l1.append(lst1[i])
            l2.append(lst2[i])
        else:
            l3.append(lst1[i])
            l4.append(lst2[i])
    explain_edges = torch.tensor(np.array([l1, l2]), dtype=torch.int64)
    complement_edges = torch.tensor(np.array([l3, l4]), dtype=torch.int64)
    return explain_edges, complement_edges

def fidelity_function(explain_edges, complement_edges, model, loader, type_of_explainer='phenomenon'):
    comp_y_hat = 0
    explain_y_hat = 0
    if type_of_explainer == 'phenomenon':
        for data in loader:  # Iterate in batches over the training/test dataset.
            data.x = torch.reshape(data.x, (data.x.shape[0], 1))
            data.x = data.x.type(torch.FloatTensor)
            data = data.to(device)
            out = model(data.x, complement_edges, data.batch)  
            pred_comp = out.argmax(dim=1)  # Use the class with highest probability.
            out = model(data.x, ei, data.batch)  
            pred_full = out.argmax(dim=1)
            out = model(data.x, explain_edges, data.batch)  
            pred_explain = out.argmax(dim=1)  # Use the class with highest probability.
            explain_y_hat += abs(int((pred_full == data.y).sum() - (pred_explain == data.y).sum()))
            comp_y_hat += abs(int((pred_full == data.y).sum() - (pred_comp == data.y).sum()))
        pos_fidelity = comp_y_hat / len(loader.dataset)
        neg_fidelity = explain_y_hat / len(loader.dataset)
    elif type_of_explainer == 'model':
        for data in loader:  # Iterate in batches over the training/test dataset.
            data.x = torch.reshape(data.x, (data.x.shape[0], 1))
            data.x = data.x.type(torch.FloatTensor)
            data = data.to(device)
            out = model(data.x, complement_edges, data.batch)  
            pred_comp = out.argmax(dim=1)  # Use the class with highest probability.
            out = model(data.x, ei, data.batch)  
            pred_full = out.argmax(dim=1)
            out = model(data.x, explain_edges, data.batch)  
            pred_explain = out.argmax(dim=1)  # Use the class with highest probability.
            comp_y_hat += int((pred_comp == pred_full).sum())
            explain_y_hat += int((pred_explain == pred_full).sum()) 
        pos_fidelity = 1 - comp_y_hat / len(loader.dataset)
        neg_fidelity = 1 - explain_y_hat / len(loader.dataset)
    else:
        pos_fidelity = comp_y_hat
        neg_fidelity = explain_y_hat
    return [pos_fidelity, neg_fidelity]

# sparsity = [25, 50, 75]
sparsity = [int(sys.argv[1])]
probs_vs_gt = []
metrics_lst = []
auprc_lst = []
from torch_geometric.explain.metric import groundtruth_metrics
for sparse in sparsity:
    adj, features, labels, num_features, num_classes = get_data(sparse)
    edge_index = torch.tensor(adj, dtype=torch.int64)
    ei = edge_index
    features = features.astype(np.float32)
    sz = np.array(adj).shape[1]
    num_edges = sz
    edge_weight = torch.ones(sz)
    num_graphs = len(labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    graph_data = DataLoader(graph_data, batch_size=1)
    print(graph_data)
    dataset = graph_data
    train_dataset = torch_geometric.data.Batch.from_data_list(train_dataset)
    test_dataset = torch_geometric.data.Batch.from_data_list(test_dataset)
    
    feat = torch.tensor(features)
    adjacency = torch.tensor(adj, dtype=torch.int64)
    y = torch.tensor(labels)
    
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
    model = GCN(hidden_channels=2, data=dataset, output_size=num_classes).to(device)
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()


    for epoch in range(1, 51):
        loss = train(model, train_loader)
        train_acc, trainAUC = test(model, train_loader)
        test_acc,testAUC = test(model, test_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Train AUC: {trainAUC:.4f}, Test Acc: {test_acc:.4f}, Test AUC: {testAUC:.4f}, Loss: {loss:.4f}')
        
    print(sparse)
    X = torch.tensor(features)
    y = torch.tensor(labels)
    # X = torch.reshape(X, (X.shape[0], X.shape[1], 1))
    G = np.load(f'sergio data/ExtraPointsSergio{sparse}.npy')
    G = torch.tensor(G, dtype=torch.int64)
    i = X.shape[1]
    
    loader = graph_data
    gt = np.load('sergio data/gt_adj.npy')
    lst1 = gt[0]
    lst2 = gt[1]
    gt = set()
    for i in range(0, len(lst1)):
        pt = (lst1[i], lst2[i])
        gt.add(pt)
    df_extra = np.load(f'sergio data/ExtraPointsSergio{sparse}.npy')
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
    l1 = np.load('sergio data/gt_adj.npy')[0]
    l2 = np.load('sergio data/gt_adj.npy')[1]
    for i in range(0, len(l1)):
        if (l1[i], l2[i]) not in full_set:
            false_negative_base += 1
    print(f'Number of Data-based FNs: {false_negative_base}')
    
    ep = 300
    eps = 25
    probs_vs_gt = []
    results = []
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    lrs = [1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.15, 0.2]
    alphas = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    betas = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    best_beta_acc = 0
    best_beta_prec = 0
    best_beta_rec = 0
    best_gnn_acc = 0
    best_gnn_prec = 0
    best_gnn_rec = 0
    for seed in seeds:
        for lr in lrs:
            set_seed(seed)
            if sys.argv[2] == 'Beta':
                for alpha in alphas:
                    for beta in betas:
                        explainer = BetaExplainer(model, graph_data, G, torch.device('cpu'), 2000, a=alpha, b=beta)
                        explainer.train(25, lr)
                        prediction_mask = explanation.edge_mask()
                        tp = 0
                        tn = 0
                        fn = false_negative_base
                        fp = 0
                        for ct in range(0, sz):
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
                        em = prediction_mask
                        met = groundtruth_metrics(em, groundtruth_mask)
                        acc = met[0]
                        rec = met[1]
                        prec = met[2]
                        bat = 'Not Applicable'
                        lst1 = [sparse, 'Beta', seed, lr, bat, alpha, beta, tp, tn, fp, fn, acc, rec, prec]
                        results.append(lst1)
                        if acc >= best_beta_acc and prec >= best_beta_prec and rec >= best_beta_rec:
                            best_beta_em = prediction_mask
                            best_beta_acc = acc
                            best_beta_prec = prec
                            best_beta_rec = rec
                            best_beta_lr = lr
                            best_alpha = alpha
                            best_beta = beta
                            best_beta_seed = seed
            elif sys.argv[2] == 'GNN':
                explainer = Explainer(
                    model=model,
                    algorithm=GNNExplainer(epochs=ep, lr=lr),
                    explanation_type='phenomenon',
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
                best_acc = 0
                best_prec = 0
                best_rec = 0
                best_em = []
                for batch in loader:
                    x = torch.reshape(batch.x, (batch.x.shape[0], 1))
                    explanation = explainer(x, batch.edge_index, target=batch.y)
                    prediction_mask = explanation.edge_mask
                    tp = 0
                    tn = 0
                    fn = false_negative_base
                    fp = 0
                    for ct in range(0, sz):
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
                    em = prediction_mask
                    met = groundtruth_metrics(em, groundtruth_mask)
                    acc = met[0]
                    rec = met[1]
                    prec = met[2]
                    if acc >= best_acc and prec >= best_prec and rec >= best_rec:
                        lst1 = [sparse, 'GNN', seed, lr, bat, 'Not Applicable', 'Not Applicable', tp, tn, fp, fn, acc, rec, prec]
                        best_acc = acc
                        best_prec = prec
                        best_rec = rec
                        best_em = em.numpy()
                        best_lr = lr
                    bat += 1
                if best_acc >= best_gnn_acc and best_prec >= best_gnn_prec and best_rec >= best_gnn_rec:
                    best_gnn_acc = best_acc
                    best_gnn_prec = best_prec
                    best_gnn_rec = best_rec
                    best_gnn_em = best_em
                    best_gnn_lr = best_lr
                    best_gnn_seed = seed
                results.append(lst1)
    gt = gt_grn.numpy()
    if sys.argv[2] == 'GNN':
        print(f'Acc: {best_gnn_acc}, Prec: {best_gnn_prec}, Rec: {best_gnn_rec}, LR: {best_gnn_lr}, Seed: {best_gnn_seed}')
        for i in range(0, best_gnn_em.shape[0]):
            probs_vs_gt.append(['GNN', best_em_gnn[i], gt[i], G[0, i], G[1, i]])
    elif sys.argv[2] == 'Beta':
        print(f'Acc: {best_beta_acc}, Prec: {best_beta_prec}, Rec: {best_beta_rec}, LR: {best_beta_lr}, Seed: {best_beta_seed}, , Alpha: {best_alpha}, Beta: {beset_beta}')
        for i in range(0, best_beta_em.shape[0]):
            probs_vs_gt.append(['Beta', best_beta_em[i], gt[i], G[0, i], G[1, i]])
    else:
        print('Not an explainer')
    
col = ['Dataset', 'Explainer', 'Seed', 'Learning Rate', 'Batch', 'Not Applicable', 'Not Applicable', 'True Positive', 'True Negative', 'False Positive', 'False Negative', 'Torch Accuracy', 'Torch Recall', 'Torch Precision']
df = pd.DataFrame(results, columns=cols)
df['Precision'] = df['TP'] / (df['TP'] + df['FP'])
df['Recall'] = df['TP'] / (df['TP'] + df['FN'])
df['Specificity'] = df['TN'] / (df['TN'] + df['FP'])
df['Accuracy'] = (df['TP'] + df['TN']) / (df['FP'] + df['FN'] + df['TP'] + df['TN'])
df['Jaccard Index'] = df['TP'] / (df['FP'] + df['FN'] + df['TP'])
df['F1 Score'] = (2 * df['Precision'] * df['Recall']) / (df['Precision'] + df['Recall'])
df.to_csv(f'Metrics{sys.argv[1]}{sys.argv[2]}.csv')
    
col = ['Dataset', 'Explainer', 'Probability of Edge Mask', 'Ground Truth?', 'Endpoint 1', 'Endpoint 2']
df = pd.DataFrame(probs_vs_gt, columns=col)
df['Category'] = df['Ground Truth?'].apply(lambda x: 'True Positive' if x == 1 else 'False Positive')
df.to_csv(f'Graphs{sys.argv[1]}{sys.argv[2]}.csv')