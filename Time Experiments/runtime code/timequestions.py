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

def get_data(data):
    x = np.load(f'{data}Data/X.npy')
    y = np.load(f'{data}Data/Y.npy')
    adj = np.load(f'{data}Data/EdgeIndex.npy')
    train_mask = np.load(f'{data}Data/TrainMask.npy')
    test_mask = np.load(f'{data}Data/TestMask.npy')
    return x, y, adj, train_mask, test_mask
    

import torch
from torch.nn import ReLU, Linear
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool


class GCN(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/1903.03894.
    This model consists of 3 stacked GCN layers followed by a linear layer.
    """
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = SAGEConv(num_features, num_classes)
        self.relu1 = ReLU()

    def forward(self, x, edge_index):
        conv1 = self.relu1(self.conv1(x, edge_index))
        return conv1
        
set_seed(0)
    
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
    
lrs = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.15, 0.2]
wds = ['', 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.15, 0.2]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results = []
source = sys.argv[1]
features, labels, adj, train_mask, test_mask = get_data(source)
classes = np.unique(labels)
num_classes = classes.shape[0]
x = torch.tensor(features, dtype=torch.float32)
edge_index = torch.tensor(adj, dtype=torch.int64)
labels = torch.tensor(labels, dtype=torch.int64)
X = x
G = edge_index
y = labels
num_features = features.shape[1]
if sys.argv[1] == 'Minesweeper':
    lr = 0.005
    wd = 0.001
elif sys.argv[1] == 'Tolokers':
    lr = 5e-5
    wd = 0.15
elif sys.argv[1] == 'Questions':
    lr = 0.001
    wd = 0.0001
else:
    lr = 0
    wd = ''
model = GCN(num_features, num_classes)
model = model.to(device)
print(model)
if wd != '':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
best_val_acc = 0.0
best_epoch = 0
for epoch in range(1, 51):
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

    
def faithfulness(model, X, G, edge_mask, bound):
    org_vec = model(X, G)
    lst = []
    for i in range(0, edge_mask.shape[0]):
        if edge_mask[i] >= bound:
            lst.append(i)
    g = G[:, lst]
    pert_vec = model(X, g)
    org_softmax = F.softmax(org_vec, dim=-1)
    pert_softmax = F.softmax(pert_vec, dim=-1)
    res = 1 - torch.exp(-F.kl_div(org_softmax.log(), pert_softmax, None, None, 'sum')).item()
    return res
    
def masked_out(G, mask, bound):
    p = []
    n = []
    for i in range(mask.shape[0]):
        if mask[i] >= bound:
            n.append(i)
        else:
            p.append(i)
    gp = torch.tensor(G[:, p], dtype=torch.int64)
    gn = torch.tensor(G[:, n], dtype=torch.int64)
    return gp, gn
    
def partial_res(a, b):
    if a == b:
        out = 1
    else:
        out = 0
    return out

def fidelity_phenomenon(model, X, G, y, mask, bound):
    gp, gn = masked_out(G.numpy(), mask, bound)
    res_p = model(X, gp).argmax(dim=1)
    res_n = model(X, gn).argmax(dim=1)
    res = model(X, G).argmax(dim=1)
    pos = 0
    neg = 0
    for i in range(y.shape[0]):
        m1 = partial_res(res[i], y[i])
        m2 = partial_res(res_p[i], y[i])
        m3 = partial_res(res_n[i], y[i])
        pos += abs(m1 - m2)
        neg += abs(m1 - m3)
    pos /= y.shape[0]
    neg /= y.shape[0]
    return pos, neg
    
def fidelity_model(model, X, G, mask, bound):
    p = 0
    n = 0
    gp, gn = masked_out(G.numpy(), mask, bound)
    res = model(X, G).argmax(dim=1)
    res_p = model(X, gp).argmax(dim=1)
    res_n = model(X, gn).argmax(dim=1)
    for i in range(0, res.shape[0]):
        p += partial_res(res_p[i], res[i])
        n += partial_res(res_n[i], res[i])
    sz = X.shape[0]
    pos = 1 - p / sz
    neg = 1 - n / sz
    return pos, neg
    
def char_score(p, n, wp=0.5, wn=0.5):
    num = (wp + wn) * p * (1 - n)
    denom = wp * (1 - p) + wn * n
    res = num / denom
    return res
    

g = G.numpy()
ru = 25
ep = 50
results = []
if sys.argv[2].lower() == 'beta':    
    if source == 'Questions':
        lr = 0.05
        alpha = 0.7
        beta = 0.7
        bound = 0.5
    else:
        lr = 0
        alpha = 0
        beta = 0
        bound = 0.5
    for run in range(0, ru):
        start = time.time()
        explainer = BetaExplainer(model, X, G, torch.device('cpu'), a=alpha, b=beta)
        explainer.train(ep, lr)
        em = explainer.edge_mask()
        end = time.time()
        runtime = end - start
        avg_runtime = runtime / ep
        print(f'Beta, Sparsity: {sys.argv[1]}, Runtime: {runtime}, Average: {(end - start) / ep}')
        results.append([sys.argv[1], 'Beta', run, ep, runtime, avg_runtime])
else:
    if source == 'Questions':
        lr = 1e-5
        ty = 'phenomenon'
        bound = 0.5
    else:
        lr = 0
        ty = 'model'
        bound = 0.5
    for run in range(0, ru):
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
        print(f'GNN, Sparsity: {sys.argv[1]}, Runtime: {runtime}, Average: {(end - start) / ep}')
        results.append([sys.argv[1], 'GNN', run, ep, runtime, avg_runtime])
df = pd.DataFrame(results, columns=['Data', 'Explainer', 'Experiment', '# Epochs', 'Full Runtime', 'Average Runtime'])
df.to_csv(f'time/Questions{sys.argv[2]}Runtimes.csv')