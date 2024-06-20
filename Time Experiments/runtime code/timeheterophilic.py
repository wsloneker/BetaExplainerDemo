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
    
# set_seed(1)
    
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
        self.embedding_size = 20 * 3
        self.conv1 = SAGEConv(num_features, num_classes)
        self.relu1 = ReLU()

    def forward(self, x, edge_index):
        input_lin = self.conv1(x, edge_index)
        relu_lin  = self.relu1(input_lin)
        return relu_lin
        
        
def get_data(data):
    '''This function allows us to load our data. We use the supergraph data - the original graph plus
        extra points associated with differnet points that are highly correlated to see if our explainers
        capture the ground truth data well'''
    labels = np.load(f'heterophilic/{data}NodeLabels.npy', allow_pickle=True)
    features = np.load(f'heterophilic/{data}Features.npy', allow_pickle=True)
    num_features = features.shape[1]
    num_classes = len(np.unique(labels))
    adj = np.load(f'heterophilic/{data}EdgeIndex.npy', allow_pickle=True)
    train_mask = np.load(f'heterophilic/{data}TrainMask.npy', allow_pickle=True)
    test_mask = np.load(f'heterophilic/{data}TestMask.npy', allow_pickle=True)
    val_mask = np.load(f'heterophilic/{data}ValMask.npy', allow_pickle=True)
    return adj, features, labels, num_features, num_classes, train_mask, test_mask, val_mask
    
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
  
source = sys.argv[1]
print(source)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results = []

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


adj, features, labels, num_features, num_classes, train_mask, test, val_mask = get_data(source)
vertex_labels = []
for label in labels:
    vertex_labels.append(label)
test_mask = []
for i in range(0, test.shape[0]):
    if test[i] == True or val_mask[i] == True:
        test_mask.append(True)
    else:
        test_mask.append(False)
features = features.astype(np.float32)

features = features.astype(np.float32)
if source == 'Texas':
    lr = 0.03
    wd = 0.09
    seed = 12
elif source == 'Wisconsin':
    lr = 0.02
    wd = 0.12
    seed = 50
else:
    lr = 0
    wd = 0
    seed = 0 

set_seed(seed)
model = GCN(num_features, num_classes)
model = model.to(device)
print(model)
if wd != '':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
x = torch.tensor(features)
edge_index = torch.tensor(adj, dtype=torch.int64)
labels = torch.tensor(labels, dtype=torch.int64)
X = x
G = edge_index
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

    # Evaluate train
    train_acc = evaluate(out[train_mask], labels[train_mask])
    test_acc = evaluate(out[test_mask], labels[test_mask])

# Train eval
train_acc = evaluate(out[train_mask], labels[train_mask])
test_acc = evaluate(out[test_mask], labels[test_mask])
print(f"final train_acc: {train_acc}, test_acc: {test_acc}")
results = []

lst = []
ru = 25
ep = 50
if sys.argv[2] == 'beta' or sys.argv[2] == 'Beta':
    if source == 'Wisconsin':
        seed = 0
        lr = 0.2
        alpha = 3.9
        beta = 3.3
        bound = 0.55
    elif source == 'Texas':
        seed = 0
        lr = 0.3
        alpha=69
        beta=70
        bound = 0.5
    else:
        seed = 0
        alpha = 0
        beta = 0
        ep = 0
        lr = 0
    set_seed(seed)
    for run in range(0, ru):
        start = time.time()
        explainer = BetaExplainer(model, X, G, torch.device('cpu'), a=alpha, b=beta)
        explainer.train(ep, lr)
        end = time.time()
        runtime = end - start
        avg_runtime = runtime / ep
        print(f'Beta, Sparsity: {sys.argv[1]}, Runtime: {runtime}, Average: {(end - start) / ep}')
        lst.append([sys.argv[1], 'Beta', run, ep, runtime, avg_runtime])

else:
    if source == 'Texas':
        seed = 20
        lr = 5e-5
        ty = 'model'
    elif source == 'Wisconsin':
        seed = 500
        lr = 1e-5
        ty = 'phenomenon'
    else:
        seed = 0
        lr = 0
        ty = 'model'
    set_seed(seed)
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
        end = time.time()
        runtime = end - start
        avg_runtime = runtime / ep
        print(f'GNN, Sparsity: {sys.argv[1]}, Runtime: {runtime}, Average: {(end - start) / ep}')
        lst.append([sys.argv[1], 'GNN', i, ep, runtime, avg_runtime])
    
        
df = pd.DataFrame(lst, columns=['Data', 'Explainer', 'Experiment', '# Epochs', 'Full Runtime', 'Average Runtime'])
df.to_csv(f'time/{sys.argv[1]}{sys.argv[2]}Runtimes.csv')