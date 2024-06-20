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
import time

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

def get_data(num, added_edges):
    '''This function allows us to load our data. We use the supergraph data - the original graph plus
        extra points associated with different points that are highly correlated to see if our explainers
        capture the ground truth data well'''
    labels = np.load(f'sergio data/SERGIOsimu_{num}Sparse_noLibEff_cTypes.npy')
    features = np.load(f'sergio data/SERGIOsimu_{num}Sparse_noLibEff_concatShuffled.npy')
    num_features = features.shape[1]
    num_classes = len(np.unique(labels))
    adj = np.load(f'sergio data/ExtraPointsSergio{num}.npy')
    num = adj.shape[1]
    num_nodes = features.shape[1]
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
    def __init__(self, model: torch.nn.Module, data: torch.Tensor, G: torch.Tensor, device: torch.device, sz, alpha=0.7, beta=0.9):
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
        self.a = alpha
        self.b = beta

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

sparse = int(sys.argv[1])
from torch_geometric.explain.metric import groundtruth_metrics
num_edges = 50
results = []
while num_edges <= 250:
    adj, features, labels, num_features, num_classes = get_data(sparse, num_edges)
    edge_index = torch.tensor(adj, dtype=torch.int64)
    G = edge_index
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
    graph_datas = torch_geometric.data.Batch.from_data_list(graph_data)
    graph_data = DataLoader(graph_datas, batch_size=2000)
    graph_data1 = DataLoader(graph_datas, batch_size=1)
    print(graph_data)
    dataset = graph_data
    train_dataset = torch_geometric.data.Batch.from_data_list(train_dataset)
    test_dataset = torch_geometric.data.Batch.from_data_list(test_dataset)
    
    feat = torch.tensor(features)
    adjacency = torch.tensor(adj, dtype=torch.int64)
    y = torch.tensor(labels)
    
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')
    
    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        
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
    i = X.shape[1]
    
    loader = graph_data
    set_seed(0)
    print('Beta!')
    ep = 50
    if sparse == 25:
        lr = 0.001
        alpha = 0.55
        beta = 0.65
    elif sparse == 50:
        lr = 0.01
        alpha = 0.5
        beta = 0.95
    else:
        lr = 0.001
        alpha = 0.55
        beta = 0.65
    start = time.time()
    explainer = BetaExplainer(model, graph_data, G, torch.device('cpu'), 2000, alpha=alpha, beta=beta)
    explainer.train(ep, lr)
    end = time.time()
    runtime = end - start
    avg_runtime = runtime / ep
    print(f'Beta, Sparsity: {sparse}, Runtime: {runtime}, Average: {(end - start) / ep}')
    results.append([sparse, 'Beta', ep, runtime, avg_runtime, G.shape[1]])
    print('GNN!')
    ep = 50
    lr = 0.00001
    start = time.time()
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
    loader = graph_data1
    bat = 0
    for batch in loader:
        if bat != 0:
            break
        else:
            x = torch.reshape(batch.x, (batch.x.shape[0], 1))
            explanation = explainer(x, G, target=batch.y)
            end = time.time()
            bat += 1
    runtime = end - start
    avg_runtime = runtime / ep
    print(f'GNN, Sparsity: {sparse}, Runtime: {runtime}, Average: {(end - start) / ep}')
    results.append([sparse, 'GNN', ep, runtime, avg_runtime, G.shape[1]])
    num_edges += 50
df = pd.DataFrame(results, columns=['Data', 'Explainer', '# Epochs', 'Full Runtime', 'Average Runtime', '# Edges'])
df.to_csv(f'graph time/SERGIO{sparse}BatchGraphRuntimes.csv')