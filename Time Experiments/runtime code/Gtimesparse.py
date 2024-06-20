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
import scipy.sparse as sp
from torch_geometric.explain.metric import groundtruth_metrics

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
    
import torch
from torch.nn import ReLU, Linear
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool


class GCN(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/1903.03894.
    This model consists of 3 stacked GCN layers followed by a linear layer.
    """
    def __init__(self, num_features, hc, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hc)
        self.conv2 = GCNConv(hc, num_classes)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.2, training=self.training)
        conv1 = self.conv1(x, edge_index)
        conv2 = self.conv2(conv1, edge_index)
        return conv2
    

def get_data(sparse, added_edges):
    folder = f'{sparse}Data/'
    x = np.load(folder + 'X' + '.npy', allow_pickle=True)
    y = np.load(folder + 'Y' + '.npy', allow_pickle=True)
    adj = np.load(folder + 'Adjacency' + '.npy', allow_pickle=True)
    num = adj.shape[1]
    num_nodes = x.shape[0]
    if added_edges <= num:
        adj = adj[:, 0:added_edges]
    else:
        adj_set = set()
        for i in range(0, adj.shape[1]):
            p1 = adj[0, i]
            p2 = adj[1, i]
            adj_set.add((p1, p1))
        while adj.shape[1] < added_edges:
            p1 = np.random.randint(0, num_nodes)
            p2 = np.random.randint(0, num_nodes)
            if (p1, p2) not in adj_set:
                b = np.array([[p1, p2]])
                adj = np.concatenate((adj, b.T), axis=1)
                adj_set.add((p1, p2))
    train = np.load(folder + 'TrainMask' + '.npy', allow_pickle=True)
    test = np.load(folder + 'TrainMask' + '.npy', allow_pickle=True)
    return x, y, adj, train, test
    
def evaluate(out, labels):
    """
    Calculates the accuracy between the prediction and the ground truth.
    :param out: predicted outputs of the explainer
    :param labels: ground truth of the data
    :returns: int accuracy
    """
    preds = out.argmax(dim=1)
    correct = preds == labels
    acc = int(correct.sum()) / int(correct.size(0))
    return acc
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
import pyro.distributions as dist
import pyro
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
import torch.distributions.constraints as constraints
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO


class BetaExplainer:
    def __init__(self, model: torch.nn.Module, X: torch.Tensor, G: torch.Tensor, device: torch.device, a=0.7, b=0.9):
        self.model = model
        self.X = X
        self.G = G
        with torch.no_grad():
            self.target = self.model(self.X, self.G).flatten()

        self.ne = G.shape[1]
        self.N = max(X.shape[1], X.shape[0], G.shape[1])
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
        preds = self.model(self.X, self.G).exp().flatten()
        with pyro.plate("data_loop"):
            pyro.sample("obs", dist.Categorical(preds), obs=ys)

    def guide(self, ys):
        alpha = pyro.param("alpha_q", self.a * torch.ones(self.N).to(self.device), constraint=constraints.positive)
        beta = pyro.param("beta_q", self.b * torch.ones(self.N).to(self.device), constraint=constraints.positive)
        alpha_edges = alpha[self.G[0, :]]
        beta_edges = beta[self.G[1, :]]
        m = pyro.sample("mask", dist.Beta(alpha_edges, beta_edges).to_event(1))
        set_masks(self.model, m, self.G, False)
        self.model(self.X, self.G).exp().flatten()

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
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def vals(prediction, gt, fnb=0):
    tp = 0
    tn = 0
    fp = 0
    fn = fnb
    for i in range(prediction.shape[0]):
        if prediction[i] >= 0.5:
            if gt[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if gt[i] == 1:
                fn += 1
            else:
                tn += 1
    return tp, tn, fp, fn
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

ep = 50
results = []
num_edges = 50
while num_edges <= 750:
    features, labels, adj, train_mask, test_mask = get_data(sys.argv[1], num_edges)
    classes = np.unique(labels)
    num_classes = classes.shape[0]
    x = torch.tensor(features, dtype=torch.float32)
    edge_index = torch.tensor(adj, dtype=torch.int64)
    labels = torch.tensor(labels, dtype=torch.int64)
    
    X = x
    G = edge_index
    seed = 0
    hc = 16
    lr = 0.01
    wd = 5e-4
    set_seed(seed)
    num_features = features.shape[1]
    model = GCN(num_features, hc, num_classes)
    model = model.to(device)
    print(model)
    if wd != '':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    best_val_acc = 0.0
    best_epoch = 0
    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], labels[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)
        optimizer.step()
    
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
    
        train_acc = evaluate(out[train_mask], labels[train_mask])
        test_acc = evaluate(out[test_mask], labels[test_mask])
    
    # Train eval
    train_acc = evaluate(out[train_mask], labels[train_mask])
    test_acc = evaluate(out[test_mask], labels[test_mask])
    print(f"final train_acc: {train_acc}, test_acc: {test_acc}, LR: {lr}, wd: {wd}")
    if sys.argv[1] == 'Cora':
        seed = 0
        lr = 1e-5
        alpha = 0.2
        beta = 0.2
        bound = 0.5
    elif sys.argv[1] == 'CiteSeer':
        seed = 0
        lr = 1e-7
        alpha = 0.05
        beta = 0.05
        bound = 0.5
    else:
        seed = 0
        lr = 0
        alpha = 0
        beta = 0
        bound = 0.5
    set_seed(seed)
    start = time.time()
    explainer = BetaExplainer(model, X, G, torch.device('cpu'), a=alpha, b=beta)
    explainer.train(ep, lr)
    em = explainer.edge_mask()
    end = time.time()
    runtime = end - start
    avg_runtime = runtime / ep
    print(f'Beta, Sparsity: {sys.argv[1]}, Runtime: {runtime}, Average: {(end - start) / ep}')
    results.append([sys.argv[1], 'Beta', ep, runtime, avg_runtime, G.shape[1]])

    if sys.argv[1] == 'Cora':
        lr = 1e-5
        seed = 3
        ty = 'model'
        bound = 0.5
    elif sys.argv[1] == 'CiteSeer':
        lr = 1e-5
        seed = 2
        ty = 'model'
        bound = 0.5
    else:
        seed = 0
        lr = 0
        ty = 'model'
        bound = 0.5
    set_seed(seed)
    start = time.time()
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=ep, lr=lr),
        explanation_type=ty,
        edge_mask_type='object',
        node_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        ),
    )
    explanation = explainer(X, G, target=labels)
    prediction_mask = explanation.edge_mask
    end = time.time()
    runtime = end - start
    avg_runtime = runtime / ep
    print(f'Beta, Sparsity: {sys.argv[1]}, Runtime: {runtime}, Average: {(end - start) / ep}')
    results.append([sys.argv[1], 'GNN', ep, runtime, avg_runtime, G.shape[1]])
    num_edges += 50

df = pd.DataFrame(results, columns=['Data', 'Explainer', '# Epochs', 'Full Runtime', 'Average Runtime', '# Edges'])
df.to_csv(f'graph time/{sys.argv[1]}GraphRuntimes.csv')