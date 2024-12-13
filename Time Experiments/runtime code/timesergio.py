import torch
import random, math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from types import MethodType
from typing import Optional, Callable, Union
from functools import partial
from torch_geometric.utils import k_hop_subgraph, to_undirected
from torch_geometric.data import Data
import matplotlib
import numpy as np
import networkx as snx
import tqdm
import torch_geometric.utils as pyg_utils
import itertools
import os.path as osp
import os
from torch.nn import PairwiseDistance as pdist
from typing import Optional  
import random
import pickle
from copy import deepcopy
from typing import Tuple
import numpy as np
import torch.nn.functional as F
from torch_geometric.utils.convert import to_networkx
from torch_geometric.data import Dataset, data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Optional, Callable, Union, Any, Tuple
from sklearn.utils.random import sample_without_replacement
import numbers
from functools import partial
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_predict
import torch_geometric
from torch_geometric.data import Data, DataLoader
import torch_geometric.transforms as Tr
from torch_geometric.nn import GCNConv, SAGEConv, ChebConv, GatedGraphConv, Linear, global_mean_pool, global_max_pool
from torch.nn import functional as F
import pandas as pd
import csv
import sklearn.metrics   
import sklearn
import sys
import ipdb
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from types import MethodType
from typing import Callable, Union
from functools import partial
from torch_geometric.utils import k_hop_subgraph, to_undirected
from torch_geometric.data import Data
import matplotlib
import tqdm
import itertools
import time
from torch_geometric.nn import GINConv
import optuna


def get_flag():
    pass

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

triangle = nx.Graph()
triangle.add_nodes_from([0, 1, 2])
triangle.add_edges_from([(0, 1), (1, 2), (2, 0)])
house = nx.house_graph()

def optimize_homophily(
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        label: torch.Tensor,
        feature_mask: torch.Tensor, 
        homophily_coef: float = 1.0, 
        epochs: int = 50, 
        connected_batch_size: int = 10,
        disconnected_batch_size: int = 10,
    ):
    '''
    Optimizes the graph features to have a set level of homophily or heterophily

    Args:
        x (torch.Tensor): Initial node features. `|V| x d` tensor, where `|V|` is number of nodes,
            `d` is dimensionality of each node feature.
        edge_index (torch.Tensor): Edge index. Standard `2 x |E|` shape.
        label (torch.Tensor): All node labels. Shape `|V|,` tensor.
        feature_mask (torch.Tensor): Boolean tensor over the dimensions of each feature. Tensor
            should be size `d,`.
        homophily_coef (float, optional): Homophily coefficient on which to optimize the level 
            of homophily or heterophily in the graph. Positive values indicate homophily while
            negative values indicate heterophily. (:default: :obj:`1.0`)
        epochs (int, optional): Number of epochs on which to optimize features. (:default: :obj:`50`)
        connected_batch_size (int, optional): Batch size at each epoch for connected nodes on which 
            to observe for the loss function. (:default: :obj:`10`) 
        disconnected_batch_size (int, optional): Batch size at each epoch for disconnected nodes on which 
            to observe for the loss function. (:default: :obj:`10`) 

    :rtype: `torch.Tensor`
    Returns:
        x (torch.Tensor): Optimized node features.
    '''

    to_opt = x.detach().clone()[:,feature_mask]

    optimizer = torch.optim.Adam([to_opt], lr=0.3)
    to_opt.requires_grad = True

    # Get indices for connected nodes having same label
    c_inds = torch.randperm(edge_index.shape[1])[:connected_batch_size]
    c_inds = c_inds[label[edge_index.t()[c_inds][:, 0]] == label[edge_index.t()[c_inds][:, 1]]]

    # Get indices for connected nodes having different label
    nc_inds = torch.randperm(edge_index.shape[1])[:connected_batch_size]
    nc_inds = nc_inds[label[edge_index.t()[nc_inds][:, 0]] != label[edge_index.t()[nc_inds][:, 1]]]

    # Get set of nodes that are either connected or not connected, with different labels:
    # [[a1, a2, a3, ...], [b1, b2, b3, ...]]
    nc_list1 = torch.full((disconnected_batch_size,), -1) # Set to dummy values in beginning
    nc_list2 = torch.full((disconnected_batch_size,), -1)

    nodes = []
    for i in range(0, x.shape[0]):
        nodes.append(i)

    for i in range(disconnected_batch_size):
        c1, c2 = random.choice(nodes), random.choice(nodes)

        # Get disconnected and with same label
        while if_edge_exists(edge_index, c1, c2) or \
                torch.any((nc_list1 == c1) & (nc_list2 == c2)) or \
                (label[c1] != label[c2]).item():
            c1, c2 = random.choice(nodes), random.choice(nodes)

        # Fill lists if we found valid choice:
        nc_list1[i] = c1
        nc_list2[i] = c2

        # May be problems with inifinite loops with large batch sizes
        #   - Should control upstream to avoid

    for i in range(epochs):
        # Compute similarities for all edges in the c_inds:
        c_cos_sim = F.cosine_similarity(to_opt[edge_index.t()[c_inds][:, 0]], to_opt[edge_index.t()[c_inds][:, 1]])
        nc_cos_sim = F.cosine_similarity(to_opt[nc_list1], to_opt[nc_list2])
        diff_label_sim = F.cosine_similarity(to_opt[edge_index.t()[nc_inds][:, 0]], to_opt[edge_index.t()[nc_inds][:, 1]])
        optimizer.zero_grad()
        loss = -homophily_coef * c_cos_sim.mean() + (homophily_coef)*(nc_cos_sim.mean() + diff_label_sim.mean())
        #loss = -homophily_coef * c_cos_sim.mean() + ((1 - homophily_coef) / 2) * (nc_cos_sim.mean() + diff_label_sim.mean())
        loss.backward()
        optimizer.step()

    # Assign to appropriate copies:
    xcopy = x.detach().clone()
    xcopy[:,feature_mask] = to_opt.detach().clone()

    return xcopy


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

device = "cuda" if torch.cuda.is_available() else "cpu"


class _BaseExplainer:
    """
    Base Class for Explainers
    """
    def __init__(self,
            model: nn.Module,
            emb_layer_name: Optional[str] = None,
            is_subgraphx: Optional[bool] = False
        ):
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
                The output of the model should be unnormalized class score.
                For example, last layer = GCNConv or Linear.
            emb_layer_name (str, optional): name of the embedding layer
                If not specified, use the last but one layer by default.
        """
        self.model = model
        self.L = len([module for module in self.model.modules()
                      if isinstance(module, MessagePassing)])
        self.explain_graph = False  # Assume node-level explanation by default
        self.subgraphx_flag = is_subgraphx
        self.__set_embedding_layer(emb_layer_name)

    def __set_embedding_layer(self, emb_layer_name: str = None):
        """
        Set the embedding layer (by default is the last but one layer).
        """
        if emb_layer_name:
            try:
                self.emb_layer = getattr(self.model, emb_layer_name)
            except AttributeError:
                raise ValueError(f'{emb_layer_name} does not exist in the model')
        else:
            self.emb_layer = list(self.model.modules())[-2]

    def _get_embedding(self, x: torch.Tensor, edge_index: torch.Tensor,
                       forward_kwargs: dict = {}):
        """
        Get the embedding.
        """
        emb = self._get_activation(self.emb_layer, x, edge_index, forward_kwargs)
        return emb

    def _set_masks(self, x: torch.Tensor, edge_index: torch.Tensor,
                   edge_mask: torch.Tensor = None, explain_feature: bool = False,
                   device = None):
        """
        Initialize the edge (and feature) masks.
        """
        (n, d), m = x.shape, edge_index.shape[1]

        # Initialize edge_mask and feature_mask for learning
        std = torch.nn.init.calculate_gain('relu') * np.sqrt(2.0 / (2 * n))
        if edge_mask is None:
            edge_mask = (torch.randn(m) * std).to(device)
            self.edge_mask = torch.nn.Parameter(edge_mask)
        else:
            self.edge_mask = torch.nn.Parameter(edge_mask)
        if explain_feature:
            feature_mask = (torch.randn(d) * 0.1).to(device)
            self.feature_mask = torch.nn.Parameter(feature_mask)

        self.loop_mask = edge_index[0] != edge_index[1]

        # Tell pytorch geometric to apply edge masks
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask
                module.__loop_mask__ = self.loop_mask

    def _clear_masks(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.edge_mask = None
        self.feature_mask = None

    def _flow(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def _predict(self, x: torch.Tensor, edge_index: torch.Tensor,
                 return_type: str = 'label', forward_kwargs: dict = {}):
        """
        Get the model's prediction.

        Args:
            x (torch.Tensor, [n x d]): node features
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            return_type (str): one of ['label', 'prob', 'log_prob']
            forward_kwargs (dict, optional): additional arguments to model.forward
                beyond x and edge_index

        Returns:
            pred (torch.Tensor, [n x ...]): model prediction
        """
        # Compute unnormalized class score
        with torch.no_grad():
            out = self.model.to(device)(x, edge_index, **forward_kwargs)
            if return_type == 'label':
                out = out.argmax(dim=-1)
            elif return_type == 'prob':
                out = F.softmax(out, dim=-1)
            elif return_type == 'log_prob':
                out = F.log_softmax(out, dim=-1)
            else:
                raise ValueError("return_type must be 'label', 'prob', or 'log_prob'")

            if self.explain_graph:
                out = out.squeeze()

            return out

    def _prob_score_func_graph(self, target_class: torch.Tensor):
        """
        Get a function that computes the predicted probability that the input graphs
        are classified as target classes.

        Args:
            target_class (int): the targeted class of the graph

        Returns:
            get_prob_score (callable): the probability score function
        """
        def get_prob_score(x: torch.Tensor,
                           edge_index: torch.Tensor,
                           forward_kwargs: dict = {}):
            prob = self._predict(x, edge_index, return_type='prob',
                                 forward_kwargs=forward_kwargs)
            score = prob[:, target_class]
            return score

        return get_prob_score

    def _prob_score_func_node(self, node_idx: torch.Tensor, target_class: torch.Tensor):
        """
        Get a function that computes the predicted probabilities that k specified nodes
        in `torch_geometric.data.Batch` (disconnected union of the input graphs)
        are classified as target classes.

        Args:
            node_idx (torch.Tensor, [k]): the indices of the k nodes interested
            target_class (torch.Tensor, [k]): the targeted classes of the k nodes

        Returns:
            get_prob_score (callable): the probability score function
        """
        if self.subgraphx_flag:
            def get_prob_score(x: torch.Tensor,
                            edge_index: torch.Tensor,
                            forward_kwargs: dict = {}):
                prob = self._predict(x, edge_index, return_type='prob',
                                    forward_kwargs=forward_kwargs)
                score = prob[node_idx, target_class]
                return score
        else:
            def get_prob_score(x: torch.Tensor,
                            edge_index: torch.Tensor,
                            forward_kwargs: dict = {}):
                prob = self._predict(x, edge_index, return_type='prob',
                                    forward_kwargs=forward_kwargs)
                score = prob[:, node_idx, target_class]
                return score

        return get_prob_score

    def _get_activation(self, layer: nn.Module, x: torch.Tensor,
                        edge_index: torch.Tensor, forward_kwargs: dict = {}):
        """
        Get the activation of the layer.
        """
        activation = {}
        def get_activation():
            def hook(model, inp, out):
                activation['layer'] = out.detach()
            return hook

        layer.register_forward_hook(get_activation())

        with torch.no_grad():
            _ = self.model(x, edge_index, **forward_kwargs)

        return activation['layer']

    def _get_k_hop_subgraph(self, node_idx: int, x: torch.Tensor,
                            edge_index: torch.Tensor, num_hops: int = None, **kwargs):
        """
        Extract the subgraph of target node

        Args:
            node_idx (int): the node index
            x (torch.Tensor, [n x d]): node feature matrix with shape
            edge_index (torch.Tensor, [2 x m]): edge index
            kwargs (dict): additional parameters of the graph

        Returns:
        """
        # TODO: use NamedTuple
        khop_info = subset, sub_edge_index, mapping, _ = \
            k_hop_subgraph(node_idx, num_hops, edge_index,
                           relabel_nodes=True, num_nodes=x.shape[0])
        return khop_info

    def get_explanation_node(self, node_idx: int,
                             x: torch.Tensor,
                             edge_index: torch.Tensor,
                             label: torch.Tensor = None,
                             num_hops: int = None,
                             forward_kwargs: dict = {}):
        """
        Explain a node prediction.

        Args:
            node_idx (int): index of the node to be explained
            x (torch.Tensor, [n x d]): node features
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            label (torch.Tensor, optional, [n x ...]): labels to explain
                If not provided, we use the output of the model.
            num_hops (int, optional): number of hops to consider
                If not provided, we use the number of graph layers of the GNN.
            forward_kwargs (dict, optional): additional arguments to model.forward
                beyond x and edge_index

        Returns:
            exp (dict):
                exp['feature_imp'] (torch.Tensor, [d]): feature mask explanation
                exp['edge_imp'] (torch.Tensor, [m]): k-hop edge importance
                exp['node_imp'] (torch.Tensor, [m]): k-hop node importance
            khop_info (4-tuple of torch.Tensor):
                0. the nodes involved in the subgraph
                1. the filtered `edge_index`
                2. the mapping from node indices in `node_idx` to their new location
                3. the `edge_index` mask indicating which edges were preserved
        """
        # If labels are needed
        label = self._predict(x, edge_index, return_type='label') if label is None else label
        # If probabilities / log probabilities are needed
        prob = self._predict(x, edge_index, return_type='prob')
        log_prob = self._predict(x, edge_index, return_type='log_prob')

        num_hops = self.L if num_hops is None else num_hops

        khop_info = subset, sub_edge_index, mapping, _ = \
            k_hop_subgraph(node_idx, num_hops, edge_index,
                           relabel_nodes=True, num_nodes=x.shape[0])
        sub_x = x[subset]

        exp = {'feature_imp': None, 'edge_imp': None}

        # Compute exp
        raise NotImplementedError()

        return exp, khop_info

    def get_explanation_graph(self, edge_index: torch.Tensor,
                              x: torch.Tensor, label: torch.Tensor,
                              forward_kwargs: dict = {}):
        """
        Explain a whole-graph prediction.

        Args:
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            x (torch.Tensor, [n x d]): node features
            label (torch.Tensor, [n x ...]): labels to explain
            forward_kwargs (dict, optional): additional arguments to model.forward
                beyond x and edge_index

        Returns:
            exp (dict):
                exp['feature_imp'] (torch.Tensor, [d]): feature mask explanation
                exp['edge_imp'] (torch.Tensor, [m]): k-hop edge importance
                exp['node_imp'] (torch.Tensor, [m]): k-hop node importance
        """
        exp = {'feature_imp': None, 'edge_imp': None}

        # Compute exp
        raise NotImplementedError()

    def get_explanation_link(self):
        """
        Explain a link prediction.
        """
        raise NotImplementedError()

EPS = 1e-15

import torch
from torch import Tensor
from functools import partial
import torch.nn.functional as F
from typing import Callable, Optional, Tuple
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data


import math
import copy
import torch
import networkx as nx
from functools import partial
from collections import Counter
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_networkx
from typing import Callable
from torch_geometric.utils.num_nodes import maybe_num_nodes
import copy
import torch
import numpy as np
from typing import Callable, Union
from scipy.special import comb
from itertools import combinations
import torch.nn.functional as F
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data, Batch, Dataset, DataLoader

'''
Code adapted from Dive into Graphs (DIG)
Code: https://github.com/divelab/DIG
'''

empty_tuple = tuple()

class MarginalSubgraphDataset(Dataset):
    """ Collect pair-wise graph data to calculate marginal contribution. """
    def __init__(self, data, exclude_mask, include_mask, subgraph_build_func):
        self.num_nodes = data.num_nodes
        self.X = data.x
        self.edge_index = data.edge_index
        self.device = self.X.device

        self.label = data.y
        self.exclude_mask = torch.tensor(exclude_mask).type(torch.float32).to(self.device)
        self.include_mask = torch.tensor(include_mask).type(torch.float32).to(self.device)
        self.subgraph_build_func = subgraph_build_func

    def __len__(self):
        return self.exclude_mask.shape[0]

    def __getitem__(self, idx):
        exclude_graph_X, exclude_graph_edge_index = self.subgraph_build_func(self.X, self.edge_index, self.exclude_mask[idx])
        include_graph_X, include_graph_edge_index = self.subgraph_build_func(self.X, self.edge_index, self.include_mask[idx])
        exclude_data = Data(x=exclude_graph_X, edge_index=exclude_graph_edge_index)
        include_data = Data(x=include_graph_X, edge_index=include_graph_edge_index)
        return exclude_data, include_data
    def len(self):
        return self.exclude_mask.shape[0]
    
    def get(self, idx):
        exclude_graph_X, exclude_graph_edge_index = self.subgraph_build_func(self.X, self.edge_index, self.exclude_mask[idx])
        include_graph_X, include_graph_edge_index = self.subgraph_build_func(self.X, self.edge_index, self.include_mask[idx])
        exclude_data = Data(x=exclude_graph_X, edge_index=exclude_graph_edge_index)
        include_data = Data(x=include_graph_X, edge_index=include_graph_edge_index)
        return exclude_data, include_data

def GnnNets_GC2value_func(gnnNets, target_class, forward_kwargs = {}):
    def value_func(batch):
        with torch.no_grad():
            #logits = gnnNets(data=batch)
            logits = gnnNets(batch.x, batch.edge_index, **forward_kwargs)
            print(forward_kwargs)
            print(batch.batch)
            probs = F.softmax(logits, dim=-1)
            print(probs, probs.shape)
            score = probs[:, target_class]
        return score
    return value_func


def GnnNets_NC2value_func(gnnNets_NC, node_idx: Union[int, torch.Tensor], target_class: torch.Tensor):
    def value_func(data):
        with torch.no_grad():
            #logits = gnnNets_NC(data=data)
            logits = gnnNets_NC(data.x, data.edge_index)
            probs = F.softmax(logits, dim=-1)
            # select the corresponding node prob through the node idx on all the sampling graphs
            batch_size = data.batch.max() + 1
            print(batch_size)
            probs = probs.reshape(batch_size, -1, probs.shape[-1])
            score = probs[:, node_idx, target_class]
            return score
    return value_func


def get_graph_build_func(build_method):
    if build_method.lower() == 'zero_filling':
        return graph_build_zero_filling
    elif build_method.lower() == 'split':
        return graph_build_split
    else:
        raise NotImplementedError


def marginal_contribution(data: Data, exclude_mask: np.ndarray, include_mask: np.ndarray,
                          value_func, subgraph_build_func):
    """ Calculate the marginal value for each pair. Here exclude_mask and include_mask are node mask. """
    marginal_subgraph_dataset = MarginalSubgraphDataset(data, exclude_mask, include_mask, subgraph_build_func)
    dataloader = DataLoader(marginal_subgraph_dataset, batch_size=256, shuffle=False, pin_memory=False, num_workers=0)

    marginal_contribution_list = []

    for exclude_data, include_data in dataloader:
        exclude_values = value_func(exclude_data)
        include_values = value_func(include_data)
        margin_values = include_values - exclude_values
        marginal_contribution_list.append(margin_values)

    marginal_contributions = torch.cat(marginal_contribution_list, dim=0)
    return marginal_contributions


def graph_build_zero_filling(X, edge_index, node_mask: torch.Tensor):
    """ subgraph building through masking the unselected nodes with zero features """
    ret_X = X * node_mask.unsqueeze(1)
    return ret_X, edge_index


def graph_build_split(X, edge_index, node_mask: torch.Tensor):
    """ subgraph building through spliting the selected nodes from the original graph """
    row, col = edge_index
    edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
    ret_edge_index = edge_index[:, edge_mask]
    return X, ret_edge_index


def l_shapley(coalition: list, data: Data, local_raduis: int,
              value_func: Callable, subgraph_building_method='zero_filling'):
    """ shapley value where players are local neighbor nodes """
    graph = to_networkx(data)
    num_nodes = graph.number_of_nodes()
    subgraph_build_func = get_graph_build_func(subgraph_building_method)

    local_region = copy.copy(coalition)
    for k in range(local_raduis - 1):
        k_neiborhoood = []
        for node in local_region:
            k_neiborhoood += list(graph.neighbors(node))
        local_region += k_neiborhoood
        local_region = list(set(local_region))

    set_exclude_masks = []
    set_include_masks = []
    nodes_around = [node for node in local_region if node not in coalition]
    num_nodes_around = len(nodes_around)

    for subset_len in range(0, num_nodes_around + 1):
        node_exclude_subsets = combinations(nodes_around, subset_len)
        for node_exclude_subset in node_exclude_subsets:
            set_exclude_mask = np.ones(num_nodes)
            set_exclude_mask[local_region] = 0.0
            if node_exclude_subset:
                set_exclude_mask[list(node_exclude_subset)] = 1.0
            set_include_mask = set_exclude_mask.copy()
            set_include_mask[coalition] = 1.0

            set_exclude_masks.append(set_exclude_mask)
            set_include_masks.append(set_include_mask)

    exclude_mask = np.stack(set_exclude_masks, axis=0)
    include_mask = np.stack(set_include_masks, axis=0)
    num_players = len(nodes_around) + 1
    num_player_in_set = num_players - 1 + len(coalition) - (1 - exclude_mask).sum(axis=1)
    p = num_players
    S = num_player_in_set
    coeffs = torch.tensor(1.0 / comb(p, S) / (p - S + 1e-6))

    marginal_contributions = \
        marginal_contribution(data, exclude_mask, include_mask, value_func, subgraph_build_func)

    l_shapley_value = (marginal_contributions.squeeze().cpu() * coeffs).sum().item()
    return l_shapley_value


def mc_shapley(coalition: list, data: Data,
               value_func: Callable, subgraph_building_method='zero_filling',
               sample_num=1000) -> float:
    """ monte carlo sampling approximation of the shapley value """
    subset_build_func = get_graph_build_func(subgraph_building_method)

    num_nodes = data.num_nodes
    node_indices = np.arange(num_nodes)
    coalition_placeholder = num_nodes
    set_exclude_masks = []
    set_include_masks = []

    for example_idx in range(sample_num):
        subset_nodes_from = [node for node in node_indices if node not in coalition]
        random_nodes_permutation = np.array(subset_nodes_from + [coalition_placeholder])
        random_nodes_permutation = np.random.permutation(random_nodes_permutation)
        split_idx = np.where(random_nodes_permutation == coalition_placeholder)[0][0]
        selected_nodes = random_nodes_permutation[:split_idx]
        set_exclude_mask = np.zeros(num_nodes)
        set_exclude_mask[selected_nodes] = 1.0
        set_include_mask = set_exclude_mask.copy()
        set_include_mask[coalition] = 1.0

        set_exclude_masks.append(set_exclude_mask)
        set_include_masks.append(set_include_mask)

    exclude_mask = np.stack(set_exclude_masks, axis=0)
    include_mask = np.stack(set_include_masks, axis=0)
    marginal_contributions = marginal_contribution(data, exclude_mask, include_mask, value_func, subset_build_func)
    mc_shapley_value = marginal_contributions.mean().item()

    return mc_shapley_value


def mc_l_shapley(coalition: list, data: Data, local_raduis: int,
                 value_func: Callable, subgraph_building_method='zero_filling',
                 sample_num=1000) -> float:
    """ monte carlo sampling approximation of the l_shapley value """
    graph = to_networkx(data)
    num_nodes = graph.number_of_nodes()
    subgraph_build_func = get_graph_build_func(subgraph_building_method)

    local_region = copy.copy(coalition)
    for k in range(local_raduis - 1):
        k_neiborhoood = []
        for node in local_region:
            k_neiborhoood += list(graph.neighbors(node))
        local_region += k_neiborhoood
        local_region = list(set(local_region))

    coalition_placeholder = num_nodes
    set_exclude_masks = []
    set_include_masks = []
    for example_idx in range(sample_num):
        subset_nodes_from = [node for node in local_region if node not in coalition]
        random_nodes_permutation = np.array(subset_nodes_from + [coalition_placeholder])
        random_nodes_permutation = np.random.permutation(random_nodes_permutation)
        split_idx = np.where(random_nodes_permutation == coalition_placeholder)[0][0]
        selected_nodes = random_nodes_permutation[:split_idx]
        set_exclude_mask = np.ones(num_nodes)
        set_exclude_mask[local_region] = 0.0
        set_exclude_mask[selected_nodes] = 1.0
        set_include_mask = set_exclude_mask.copy()
        set_include_mask[coalition] = 1.0

        set_exclude_masks.append(set_exclude_mask)
        set_include_masks.append(set_include_mask)

    exclude_mask = np.stack(set_exclude_masks, axis=0)
    include_mask = np.stack(set_include_masks, axis=0)
    marginal_contributions = \
        marginal_contribution(data, exclude_mask, include_mask, value_func, subgraph_build_func)

    mc_l_shapley_value = (marginal_contributions).mean().item()
    return mc_l_shapley_value


def gnn_score(coalition: list, data: Data, value_func: Callable,
              subgraph_building_method='zero_filling') -> torch.Tensor:
    """ the value of subgraph with selected nodes """
    num_nodes = data.num_nodes
    subgraph_build_func = get_graph_build_func(subgraph_building_method)
    mask = torch.zeros(num_nodes).type(torch.float32).to(data.x.device)
    mask[coalition] = 1.0
    ret_x, ret_edge_index = subgraph_build_func(data.x, data.edge_index, mask)
    mask_data = Data(x=ret_x, edge_index=ret_edge_index)
    mask_data = Batch.from_data_list([mask_data])
    score = value_func(mask_data)
    # get the score of predicted class for graph or specific node idx
    return score.item()


def NC_mc_l_shapley(coalition: list, data: Data, local_raduis: int,
                    value_func: Callable, node_idx: int=-1, subgraph_building_method='zero_filling', sample_num=1000) -> float:
    """ monte carlo approximation of l_shapley where the target node is kept in both subgraph """
    graph = to_networkx(data)
    num_nodes = graph.number_of_nodes()
    subgraph_build_func = get_graph_build_func(subgraph_building_method)

    local_region = copy.copy(coalition)
    for k in range(local_raduis - 1):
        k_neiborhoood = []
        for node in local_region:
            k_neiborhoood += list(graph.neighbors(node))
        local_region += k_neiborhoood
        local_region = list(set(local_region))

    coalition_placeholder = num_nodes
    set_exclude_masks = []
    set_include_masks = []
    for example_idx in range(sample_num):
        subset_nodes_from = [node for node in local_region if node not in coalition]
        random_nodes_permutation = np.array(subset_nodes_from + [coalition_placeholder])
        random_nodes_permutation = np.random.permutation(random_nodes_permutation)
        split_idx = np.where(random_nodes_permutation == coalition_placeholder)[0][0]
        selected_nodes = random_nodes_permutation[:split_idx]
        set_exclude_mask = np.ones(num_nodes)
        set_exclude_mask[local_region] = 0.0
        set_exclude_mask[selected_nodes] = 1.0
        if node_idx != -1:
            set_exclude_mask[node_idx] = 1.0
        set_include_mask = set_exclude_mask.copy()
        set_include_mask[coalition] = 1.0  # include the node_idx

        set_exclude_masks.append(set_exclude_mask)
        set_include_masks.append(set_include_mask)

    exclude_mask = np.stack(set_exclude_masks, axis=0)
    include_mask = np.stack(set_include_masks, axis=0)
    marginal_contributions = \
        marginal_contribution(data, exclude_mask, include_mask, value_func, subgraph_build_func)

    mc_l_shapley_value = (marginal_contributions).mean().item()
    return mc_l_shapley_value

'''
Code adapted from Dive into Graphs (DIG)
Code: https://github.com/divelab/DIG
'''

def find_closest_node_result(results, max_nodes):
    """ return the highest reward tree_node with its subgraph is smaller than max_nodes """

    results = sorted(results, key=lambda x: len(x.coalition))

    result_node = results[0]
    for result_idx in range(len(results)):
        x = results[result_idx]
        if len(x.coalition) <= max_nodes and x.P > result_node.P:
            result_node = x
    return result_node


def reward_func(reward_method, value_func, node_idx=None,
                local_radius=4, sample_num=100,
                subgraph_building_method='zero_filling'):
    if reward_method.lower() == 'gnn_score':
        return partial(gnn_score,
                       value_func=value_func,
                       subgraph_building_method=subgraph_building_method)

    elif reward_method.lower() == 'mc_shapley':
        return partial(mc_shapley,
                       value_func=value_func,
                       subgraph_building_method=subgraph_building_method,
                       sample_num=sample_num)

    elif reward_method.lower() == 'l_shapley':
        return partial(l_shapley,
                       local_raduis=local_radius,
                       value_func=value_func,
                       subgraph_building_method=subgraph_building_method)

    elif reward_method.lower() == 'mc_l_shapley':
        return partial(mc_l_shapley,
                       local_raduis=local_radius,
                       value_func=value_func,
                       subgraph_building_method=subgraph_building_method,
                       sample_num=sample_num)

    elif reward_method.lower() == 'nc_mc_l_shapley':
        assert node_idx is not None, " Wrong node idx input "
        return partial(NC_mc_l_shapley,
                       node_idx=node_idx,
                       local_raduis=local_radius,
                       value_func=value_func,
                       subgraph_building_method=subgraph_building_method,
                       sample_num=sample_num)

    else:
        raise NotImplementedError


def k_hop_subgraph_with_default_whole_graph(
        edge_index, node_idx=None, num_hops=3, relabel_nodes=False,
        num_nodes=None, flow='source_to_target'):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.
    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index  # edge_index 0 to 1, col: source, row: target

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    inv = None

    if node_idx is None:
        subsets = torch.tensor([0])
        cur_subsets = subsets
        while 1:
            node_mask.fill_(False)
            node_mask[subsets] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets = torch.cat([subsets, col[edge_mask]]).unique()
            if not cur_subsets.equal(subsets):
                cur_subsets = subsets
            else:
                subset = subsets
                break
    else:
        if isinstance(node_idx, (int, list, tuple)):
            node_idx = torch.tensor([node_idx], device=row.device, dtype=torch.int64).flatten()
        elif isinstance(node_idx, torch.Tensor) and len(node_idx.shape) == 0:
            node_idx = torch.tensor([node_idx], device=row.device)
        else:
            node_idx = node_idx.to(row.device)

        subsets = [node_idx]
        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets.append(col[edge_mask])
        subset, inv = torch.cat(subsets).unique(return_inverse=True)
        inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes,), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask  # subset: key new node idx; value original node idx


def compute_scores(score_func, children):
    results = []
    for child in children:
        if child.P == 0:
            score = score_func(child.coalition, child.data)
        else:
            score = child.P
        results.append(score)
    return results

class MCTSNode(object):

    def __init__(self, coalition: list, data: Data, ori_graph: nx.Graph,
                 c_puct: float = 10.0, W: float = 0, N: int = 0, P: float = 0,
                 mapping = None):
        self.data = data
        self.coalition = coalition # Coalition of possible subsets of players
        self.ori_graph = ori_graph # Original input graph
        self.c_puct = c_puct # Hyperparameter in search algorithm
        self.children = [] # Children within MCTS tree
        self.W = W  # sum of node value
        self.N = N  # times of arrival
        self.P = P  # property score (reward)

        self.mapping = mapping # ADDED from OWEN

    def Q(self): # Average of W
        return self.W / self.N if self.N > 0 else 0

    def U(self, n): # Action selection criteria for MCTS
        return self.c_puct * self.P * math.sqrt(n) / (1 + self.N)


class MCTS(object):
    r"""
    Monte Carlo Tree Search Method

    Args:
        X (:obj:`torch.Tensor`): Input node features
        edge_index (:obj:`torch.Tensor`): The edge indices.
        num_hops (:obj:`int`): The number of hops :math:`k`.
        n_rollout (:obj:`int`): The number of sequence to build the monte carlo tree.
        min_atoms (:obj:`int`): The number of atoms for the subgraph in the monte carlo tree leaf node.
        c_puct (:obj:`float`): The hyper-parameter to encourage exploration while searching.
        expand_atoms (:obj:`int`): The number of children to expand.
        high2low (:obj:`bool`): Whether to expand children tree node from high degree nodes to low degree nodes.
        node_idx (:obj:`int`): The target node index to extract the neighborhood.
        score_func (:obj:`Callable`): The reward function for tree node, such as mc_shapely and mc_l_shapely.

    """
    def __init__(self, X: torch.Tensor, edge_index: torch.Tensor, num_hops: int,
                 n_rollout: int = 10, min_atoms: int = 3, c_puct: float = 10.0,
                 expand_atoms: int = 14, high2low: bool = False,
                 node_idx: int = None, score_func: Callable = None):

        self.X = X
        self.edge_index = edge_index
        self.num_hops = num_hops
        self.data = Data(x=self.X, edge_index=self.edge_index)
        self.graph = to_networkx(self.data, to_undirected=True) # NETWORKX VERSION OF GRAPH
        self.data = Batch.from_data_list([self.data])
        self.num_nodes = self.graph.number_of_nodes()
        self.score_func = score_func
        self.n_rollout = n_rollout
        self.min_atoms = min_atoms
        self.c_puct = c_puct
        self.expand_atoms = expand_atoms
        self.high2low = high2low

        inv_mapping = None

        # extract the sub-graph and change the node indices.
        if node_idx is not None:
            self.ori_node_idx = node_idx
            self.ori_graph = copy.copy(self.graph)
            x, edge_index, subset, edge_mask, kwargs = \
                self.__subgraph__(node_idx, self.X, self.edge_index, self.num_hops)
            self.data = Batch.from_data_list([Data(x=x, edge_index=edge_index)])
            self.graph = self.ori_graph.subgraph(subset.tolist())
            mapping = {int(v): k for k, v in enumerate(subset)}
            inv_mapping = {v:k for k, v in mapping.items()}
            self.graph = nx.relabel_nodes(self.graph, mapping)
            self.node_idx = torch.where(subset == self.ori_node_idx)[0]
            self.num_nodes = self.graph.number_of_nodes()
            self.subset = subset

        self.root_coalition = sorted([node for node in range(self.num_nodes)])
        self.MCTSNodeClass = partial(MCTSNode, data=self.data, ori_graph=self.graph, c_puct=self.c_puct, mapping = inv_mapping)
        self.root = self.MCTSNodeClass(self.root_coalition) # Root of tree
        self.state_map = {str(self.root.coalition): self.root}

    def set_score_func(self, score_func):
        self.score_func = score_func

    @staticmethod
    def __subgraph__(node_idx, x, edge_index, num_hops, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)
        subset, edge_index, _, edge_mask = k_hop_subgraph_with_default_whole_graph(
            edge_index, node_idx, num_hops, relabel_nodes=True, num_nodes=num_nodes)

        x = x[subset]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, edge_index, subset, edge_mask, kwargs

    def mcts_rollout(self, tree_node):
        cur_graph_coalition = tree_node.coalition
        if len(cur_graph_coalition) <= self.min_atoms:
            return tree_node.P

        # Expand if this node has never been visited
        if len(tree_node.children) == 0:
            node_degree_list = list(self.graph.subgraph(cur_graph_coalition).degree)
            node_degree_list = sorted(node_degree_list, key=lambda x: x[1], reverse=self.high2low)
            all_nodes = [x[0] for x in node_degree_list]

            if len(all_nodes) < self.expand_atoms:
                expand_nodes = all_nodes
            else:
                expand_nodes = all_nodes[:self.expand_atoms]

            for each_node in expand_nodes:
                # for each node, pruning it and get the remaining sub-graph
                # here we check the resulting sub-graphs and only keep the largest one
                subgraph_coalition = [node for node in all_nodes if node != each_node]

                subgraphs = [self.graph.subgraph(c)
                             for c in nx.connected_components(self.graph.subgraph(subgraph_coalition))]
                main_sub = subgraphs[0]
                for sub in subgraphs:
                    if sub.number_of_nodes() > main_sub.number_of_nodes():
                        main_sub = sub

                new_graph_coalition = sorted(list(main_sub.nodes()))

                # check the state map and merge the same sub-graph
                Find_same = False
                for old_graph_node in self.state_map.values():
                    if Counter(old_graph_node.coalition) == Counter(new_graph_coalition):
                        new_node = old_graph_node
                        Find_same = True

                if Find_same == False:
                    new_node = self.MCTSNodeClass(new_graph_coalition)
                    self.state_map[str(new_graph_coalition)] = new_node

                Find_same_child = False
                for cur_child in tree_node.children:
                    if Counter(cur_child.coalition) == Counter(new_graph_coalition):
                        Find_same_child = True

                if Find_same_child == False:
                    tree_node.children.append(new_node)

            scores = compute_scores(self.score_func, tree_node.children)
            for child, score in zip(tree_node.children, scores):
                child.P = score

        sum_count = sum([c.N for c in tree_node.children])
        selected_node = max(tree_node.children, key=lambda x: x.Q() + x.U(sum_count))
        v = self.mcts_rollout(selected_node)
        selected_node.W += v
        selected_node.N += 1
        return v

    def mcts(self, verbose=True):

        if verbose:
            print(f"The nodes in graph is {self.graph.number_of_nodes()}")
        for rollout_idx in range(self.n_rollout):
            self.mcts_rollout(self.root)
            if verbose:
                print(f"At the {rollout_idx} rollout, {len(self.state_map)} states that have been explored.")

        explanations = [node for _, node in self.state_map.items()]
        explanations = sorted(explanations, key=lambda x: x.P, reverse=True)
        # Sorts explanations based on P value (i.e. Score(.,.,.) function in MCTS)
        return explanations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph

device = "cuda" if torch.cuda.is_available() else "cpu"

class SubgraphX(_BaseExplainer):
    r"""
    Code adapted from Dive into Graphs (DIG)
    Code: https://github.com/divelab/DIG

    The implementation of paper
    `On Explainability of Graph Neural Networks via Subgraph Explorations <https://arxiv.org/abs/2102.05152>`_.

    Args:
        model (:obj:`torch.nn.Module`): The target model prepared to explain
        num_hops(:obj:`int`, :obj:`None`): The number of hops to extract neighborhood of target node
          (default: :obj:`None`)
        rollout(:obj:`int`): Number of iteration to get the prediction
        min_atoms(:obj:`int`): Number of atoms of the leaf node in search tree
        c_puct(:obj:`float`): The hyperparameter which encourages the exploration
        expand_atoms(:obj:`int`): The number of atoms to expand
          when extend the child nodes in the search tree
        high2low(:obj:`bool`): Whether to expand children nodes from high degree to low degree when
          extend the child nodes in the search tree (default: :obj:`False`)
        local_radius(:obj:`int`): Number of local radius to calculate :obj:`l_shapley`, :obj:`mc_l_shapley`
        sample_num(:obj:`int`): Sampling time of monte carlo sampling approximation for
          :obj:`mc_shapley`, :obj:`mc_l_shapley` (default: :obj:`mc_l_shapley`)
        reward_method(:obj:`str`): The command string to select the
        subgraph_building_method(:obj:`str`): The command string for different subgraph building method,
          such as :obj:`zero_filling`, :obj:`split` (default: :obj:`zero_filling`)

    Example:
        >>> # For graph classification task
        >>> subgraphx = SubgraphX(model=model, num_classes=2)
        >>> _, explanation_results, related_preds = subgraphx(x, edge_index)

    """
    def __init__(self, model, num_hops: Optional[int] = None,
                 rollout: int = 10, min_atoms: int = 3, c_puct: float = 10.0, expand_atoms=14,
                 high2low=False, local_radius=4, sample_num=100, reward_method='mc_l_shapley',
                 subgraph_building_method='zero_filling'):

        super().__init__(model=model, is_subgraphx=True)
        self.model.eval()
        self.num_hops = self.update_num_hops(num_hops)

        # mcts hyper-parameters
        self.rollout = rollout
        self.min_atoms = min_atoms # N_{min}
        self.c_puct = c_puct
        self.expand_atoms = expand_atoms
        self.high2low = high2low

        # reward function hyper-parameters
        self.local_radius = local_radius
        self.sample_num = sample_num
        self.reward_method = reward_method
        self.subgraph_building_method = subgraph_building_method

    def update_num_hops(self, num_hops):
        if num_hops is not None:
            return num_hops

        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def get_reward_func(self, value_func, node_idx=None, explain_graph = False):
        if explain_graph:
            node_idx = None
        else:
            assert node_idx is not None
        return reward_func(reward_method=self.reward_method,
                           value_func=value_func,
                           node_idx=node_idx,
                           local_radius=self.local_radius,
                           sample_num=self.sample_num,
                           subgraph_building_method=self.subgraph_building_method)

    def get_mcts_class(self, x, edge_index, node_idx: int = None, score_func: Callable = None, explain_graph = False):
        if explain_graph:
            node_idx = None
        else:
            assert node_idx is not None
        return MCTS(x, edge_index,
                    node_idx=node_idx,
                    score_func=score_func,
                    num_hops=self.num_hops,
                    n_rollout=self.rollout,
                    min_atoms=self.min_atoms,
                    c_puct=self.c_puct,
                    expand_atoms=self.expand_atoms,
                    high2low=self.high2low)

    def get_explanation_node(self, 
            x: Tensor, 
            edge_index: Tensor, 
            node_idx: int, 
            label: int = None, 
            y = None,
            max_nodes: int = 14, 
            forward_kwargs: dict = {}
        ) -> Tuple[dict, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        '''
        Get explanation for a single node within a graph.
        Args:
            x (torch.Tensor): Input features for every node in the graph.
            edge_index (torch.Tensor): Edge index for entire input graph.
            node_idx (int): Node index for which to generate an explanation.
            label (int, optional): Label for which to assume as a prediction from 
                the model when generating an explanation. If `None`, this argument 
                is set to the prediction directly from the model. (default: :obj:`None`)
            max_nodes (int, optional): Maximum number of nodes to include in the subgraph 
                generated from the explanation. (default: :obj:`14`)
            forward_kwargs (dict, optional): Additional arguments to model.forward 
                beyond x and edge_index. Must be keyed on argument name. 
                (default: :obj:`{}`)

        :rtype: :class:`Explanation`
        Returns:
            exp (dict):
                exp['feature_imp'] is `None` because no feature explanations are generated.
                exp['node_imp'] (torch.Tensor, (n,)): Node mask of size `(n,)` where `n` 
                    is number of nodes in the entire graph described by `edge_index`. 
                    Type is `torch.bool`, with `True` indices corresponding to nodes 
                    included in the subgraph.
                exp['edge_imp'] (torch.Tensor, (e,)): Edge mask of size `(e,)` where `e` 
                    is number of edges in the entire graph described by `edge_index`. 
                    Type is `torch.bool`, with `True` indices corresponding to edges 
                    included in the subgraph.
            khop_info (4-tuple of torch.Tensor):
                0. the nodes involved in the subgraph
                1. the filtered `edge_index`
                2. the mapping from node indices in `node_idx` to their new location
                3. the `edge_index` mask indicating which edges were preserved 
        '''

        if y is not None:
            label = y[node_idx] # Get node_idx of label

        if label is None:
            self.model.eval()
            x = torch.reshape(x, (x.shape[0], 1))
            pred = self.model(x.to(device), edge_index.to(device), **forward_kwargs)
            label = int(pred.argmax(dim=1).item())
        else:
            label = int(label)

        # collect all the class index
        logits = self.model(x.to(device), edge_index.to(device), **forward_kwargs)
        probs = F.softmax(logits, dim=-1)
        probs = probs.squeeze()

        prediction = probs[node_idx].argmax(-1)
        self.mcts_state_map = self.get_mcts_class(x, edge_index, node_idx=node_idx)
        self.node_idx = self.mcts_state_map.node_idx
        # mcts will extract the subgraph and relabel the nodes
        # value_func = GnnNets_NC2value_func(self.model,
        #                                     node_idx=self.mcts_state_map.node_idx,
        #                                     target_class=label)
        value_func = self._prob_score_func_node(
            node_idx = self.mcts_state_map.node_idx,
            target_class = label
        )
        #value_func = partial(value_func, forward_kwargs=forward_kwargs)
        def wrap_value_func(data):
            return value_func(x=data.x.to(device), edge_index=data.edge_index.to(device), forward_kwargs=forward_kwargs)

        payoff_func = self.get_reward_func(wrap_value_func, node_idx=self.mcts_state_map.node_idx, explain_graph = False)
        self.mcts_state_map.set_score_func(payoff_func)
        results = self.mcts_state_map.mcts(verbose=False)

        # Get best result that has less than max nodes:
        best_result = find_closest_node_result(results, max_nodes=max_nodes)

        # Need to parse results:
        node_mask, edge_mask = self.__parse_results(best_result, edge_index)

        #print('args', node_idx, self.L, edge_index)
        khop_info = k_hop_subgraph(node_idx, self.L, edge_index)
        subgraph_edge_mask = khop_info[3] # Mask over edges

        # Set explanation
        # exp = Explanation(
        #     node_imp = 1*node_mask[khop_info[0]], # Apply node mask
        #     edge_imp = 1*edge_mask[subgraph_edge_mask],
        #     node_idx = node_idx
        # )
        exp = Explanation(
            node_imp = 1*node_mask, # Apply node mask
            edge_imp = 1*edge_mask,
            node_idx = node_idx
        )

        exp.set_enclosing_subgraph(khop_info)

        return exp

    def get_explanation_graph(self, 
            x: Tensor, 
            edge_index: Tensor, 
            label: int = None, 
            max_nodes: int = 14, 
            forward_kwargs: dict = {}, 
        ):
        '''
        Get explanation for a whole graph prediction.
        Args:
            x (torch.Tensor): Input features for every node in the graph.
            edge_index (torch.Tensor): Edge index for entire input graph.
            label (int, optional): Label for which to assume as a prediction from 
                the model when generating an explanation. If `None`, this argument 
                is set to the prediction directly from the model. (default: :obj:`None`)
            max_nodes (int, optional): Maximum number of nodes to include in the subgraph 
                generated from the explanation. (default: :obj:`14`)
            forward_kwargs (dict, optional): Additional arguments to model.forward 
                beyond x and edge_index. Must be keyed on argument name. 
                (default: :obj:`{}`)

        :rtype: :class:`Explanation`
        Returns:
            exp (dict):
                exp['feature_imp'] is `None` because no feature explanations are generated.
                exp['node_imp'] (torch.Tensor, (n,)): Node mask of size `(n,)` where `n` 
                    is number of nodes in the entire graph described by `edge_index`. 
                    Type is `torch.bool`, with `True` indices corresponding to nodes 
                    included in the subgraph.
                exp['edge_imp'] (torch.Tensor, (e,)): Edge mask of size `(e,)` where `e` 
                    is number of edges in the entire graph described by `edge_index`. 
                    Type is `torch.bool`, with `True` indices corresponding to edges 
                    included in the subgraph.
        '''
        self.model.eval()
        x = torch.reshape(x, (x.shape[0], 1))
        pred = self.model(x, edge_index, **forward_kwargs).argmax(dim=1)
        #label = int(pred.argmax(dim=1).item())
        # collect all the class index
        logits = self.model(x, edge_index, **forward_kwargs)
        
        probs = F.softmax(logits, dim=-1)
        probs = probs.squeeze()

        prediction = probs.argmax(-1)
        value_func = self._prob_score_func_graph(target_class = label)
        def wrap_value_func(data):
            return value_func(x=data.x, edge_index=data.edge_index, forward_kwargs=forward_kwargs)

        payoff_func = self.get_reward_func(wrap_value_func, explain_graph = True)
        self.mcts_state_map = self.get_mcts_class(x, edge_index, score_func=payoff_func, explain_graph = True)
        results = self.mcts_state_map.mcts(verbose=False)
        best_result = find_closest_node_result(results, max_nodes=max_nodes)

        node_mask, edge_mask = self.__parse_results(best_result, edge_index)
        exp = Explanation(
            node_imp = node_mask.float(),
            edge_imp = edge_mask.float()
        )
        # exp.node_imp = node_mask
        # exp.edge_imp = edge_mask
        exp.set_whole_graph(Data(x=x, edge_index=edge_index))

        #return {'feature_imp': None, 'node_imp': node_mask, 'edge_imp': edge_mask}
        return exp

    def __parse_results(self, best_subgraph, edge_index):
        # Function strongly based on torch_geometric.utils.subgraph function
        # Citation: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/subgraph.html#subgraph

        # Get mapping
        map = best_subgraph.mapping

        all_nodes = torch.unique(edge_index)

        subgraph_nodes = torch.tensor([map[c] for c in best_subgraph.coalition], dtype=torch.long) if map is not None \
            else torch.tensor(best_subgraph.coalition, dtype=torch.long)

        # Create node mask:
        node_mask = torch.zeros(all_nodes.shape, dtype=torch.bool)
        #node_mask[subgraph_nodes] = 1

        # Create edge_index mask
        num_nodes = maybe_num_nodes(edge_index)
        n_mask = torch.zeros(num_nodes, dtype = torch.bool)
        n_mask[subgraph_nodes] = 1

        edge_mask = n_mask[edge_index[0]] & n_mask[edge_index[1]]
        return node_mask, edge_mask


    
def graph_exp_acc(gt_exp, generated_exp, node_thresh_factor = 0.5) -> float:
    '''
    Args:
        gt_exp (Explanation): Ground truth explanation from the dataset.
        generated_exp (Explanation): Explanation output by an explainer.
    '''
    EPS = 1e-09
    JAC_feat = None
    JAC_node = None
    JAC_edge = None

    JAC_edge = []
    prec_edge = []
    rec_edge = []
    TPs = []
    FPs = []
    FNs = []
    true_edges = torch.where(gt_exp.edge_imp == 1)[0]
    for edge in range(gt_exp.edge_imp.shape[0]):
        if generated_exp[edge] >= node_thresh_factor:
            if edge in true_edges:
                TPs.append(edge)
            else:
                FPs.append(edge)
        else:
            if edge in true_edges:
                FNs.append(edge)
    TP = len(TPs)
    FP = len(FPs)
    FN = len(FNs)
    JAC = TP / (TP + FP + FN + EPS)
    prec = TP / (TP + FP + EPS)
    rec = TP / (TP + FN + EPS)
    num = (2 * prec * rec)
    if num == 0:
        F1 = 0
    else:
        F1 = num / (prec + rec)
    return JAC, prec, rec, F1
    
def to_networkx_conv(data, node_attrs=None, edge_attrs=None, to_undirected=False,
                remove_self_loops=False, get_map = False):
    r"""Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`networkx.Graph` if :attr:`to_undir
    ected` is set to :obj:`True`, or
    a directed :obj:`networkx.DiGraph` otherwise.

    Args:
        data (torch_geometric.data.Data): The data object.
        node_attrs (iterable of str, optional): The node attributes to be
            copied. (default: :obj:`None`)
        edge_attrs (iterable of str, optional): The edge attributes to be
            copied. (default: :obj:`None`)
        to_undirected (bool, optional): If set to :obj:`True`, will return a
            a :obj:`networkx.Graph` instead of a :obj:`networkx.DiGraph`. The
            undirected graph will correspond to the upper triangle of the
            corresponding adjacency matrix. (default: :obj:`False`)
        remove_self_loops (bool, optional): If set to :obj:`True`, will not
            include self loops in the resulting graph. (default: :obj:`False`)
        get_map (bool, optional): If `True`, returns a tuple where the second
            element is a map from original node indices to new ones.
            (default: :obj:`False`)
    """
    if to_undirected:
        G = nx.Graph()
        #data.edge_index = pyg_utils.to_undirected(data.edge_index)
    else:
        G = nx.DiGraph()

    node_list = sorted(torch.unique(data.edge_index).tolist())
    #node_list = np.arange(data.x.shape[0])
    map_norm = {node_list[i]:i for i in range(len(node_list))}
    rev_map_norm = {v:k for k, v in map_norm.items()}
    G.add_nodes_from([map_norm[n] for n in node_list])

    values = {}
    for key, item in data:
        if torch.is_tensor(item):
            values[key] = item.squeeze().tolist()
        else:
            values[key] = item
        if isinstance(values[key], (list, tuple)) and len(values[key]) == 1:
            values[key] = item[0]

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):
        u = map_norm[u]
        v = map_norm[v]

        if to_undirected and v > u:
            continue

        if remove_self_loops and u == v:
            continue

        G.add_edge(u, v)
        for key in edge_attrs if edge_attrs is not None else []:
            G[u][v][key] = values[key][i]

    for key in node_attrs if node_attrs is not None else []:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})

    if get_map:
        return G, map_norm
    else:
        G = nx.relabel_nodes(G, mapping=rev_map_norm)
        return G

def mask_graph(edge_index: torch.Tensor, 
        node_mask: torch.Tensor = None, 
        edge_mask: torch.Tensor = None):
    '''
    Masks the edge_index of a graph given either node_mask or edge_mask
    Args:
        edge_index (torch.tensor, dtype=torch.int)
        node_mask (torch.tensor, dtype=bool)
        edge_mask (torch.tensor, dtype=bool)
    '''
    # edge_index's are always size (2,e) with e=number of edges
    if node_mask is not None:
        nodes = node_mask.nonzero(as_tuple=True)[0].tolist()
        created_edge_mask = torch.zeros(edge_index.shape[1])
        for i in range(edge_index.shape[1]):
            edge = edge_index[:,i]
            if (edge[0] in nodes) or (edge[1] in nodes):
                created_edge_mask[i] = 1
        created_edge_mask = created_edge_mask.type(bool)
        edge_index = edge_index[:,created_edge_mask]
    elif edge_mask is not None:
        edge_index = edge_index[:,edge_mask]
    return edge_index

def whole_graph_mask_to_subgraph(node_mask, edge_mask = None, subgraph_nodes = None, subgraph_eidx = None):
    '''Converts mask of whole graph to a mask of a subgraph'''
    nodes = node_mask.nonzero(as_tuple=True)[0]
    subgraph_node_mask = torch.tensor([n.item() in nodes.tolist() for n in subgraph_nodes], dtype = torch.bool) \
            if subgraph_nodes is not None else None
    return subgraph_node_mask, None

def khop_subgraph_nx(
        node_idx: int,
        num_hops: int, 
        G: nx.Graph
    ):
    '''
    Finds k-hop neighborhood in a networkx graph. Uses a BFS of depth num_hops
        on the networkx Graph provided to find edges.
    ..note:: Includes node_idx within subgraph
    Args:
        node_idx (int): Node for which we are to find a subgraph around.
        num_hops (int): Number of hops for which to search.
        G (nx.Graph): Graph on which to find k-hop subgraph
    :rtype: list
        nodes (list): Nodes in the k-hop subgraph
    '''
    edges = list(nx.bfs_edges(G, node_idx, depth_limit = num_hops))
    return list(np.unique(edges))

 
from torch_geometric.utils import from_networkx, k_hop_subgraph, subgraph
from torch_geometric.data import Data   
def match_torch_to_nx_edges(G: nx.Graph, edge_index: torch.Tensor):
    '''
    Gives dictionary matching index in edge_index to G.edges
        - Supports matching for undirected edges
        - Mainly for plotting
    '''

    edges_list = list(G.edges)

    edges_map = dict()

    for i in range(len(edges_list)):
        e1, e2 = edges_list[i]

        # Check e1 -> 0, e2 -> 1
        # cond1 = ((e1 == edge_index[0,:]) & (e2 == edge_index[1,:])).nonzero(as_tuple=True)[0]
        # cond2 = ((e2 == edge_index[0,:]) & (e1 == edge_index[1,:])).nonzero(as_tuple=True)[0]
        cond1 = ((e1 == edge_index[0,:]) & (e2 == edge_index[1,:])).nonzero(as_tuple=True)[0]
        cond2 = ((e2 == edge_index[0,:]) & (e1 == edge_index[1,:])).nonzero(as_tuple=True)[0]
        #print(cond1)

        if cond1.shape[0] > 0:
            edges_map[(e1, e2)] = cond1[0].item()
            edges_map[(e2, e1)] = cond1[0].item()
        elif cond2.shape[0] > 0:
            edges_map[(e1, e2)] = cond2[0].item()
            edges_map[(e2, e1)] = cond2[0].item()
        else:
            raise ValueError('Edge not in graph')
    return edges_map

def remove_duplicate_edges(edge_index):
    # Removes duplicate edges from edge_index, making it arbitrarily directed (random positioning):

    new_edge_index = []
    added_nodes = set()
    dict_tracker = dict()

    edge_mask = torch.zeros(edge_index.shape[1], dtype=bool)

    for i in range(edge_index.shape[1]):
        e1 = edge_index[0,i].item()
        e2 = edge_index[1,i].item()
        if e1 in added_nodes:
            if (e2 in dict_tracker[e1]):
                continue
            dict_tracker[e1].append(e2)
        else:
            dict_tracker[e1] = [e2]
            added_nodes.add(e1)
        if e2 in added_nodes:
            if (e1 in dict_tracker[e2]):
                continue
            dict_tracker[e2].append(e1)
        else:
            dict_tracker[e2] = [e1]
            added_nodes.add(e2)

        new_edge_index.append((e1, e2)) # Append only one version
        edge_mask[i] = True
    return torch.tensor(new_edge_index).t().contiguous(), edge_mask
    
from torch.nn import PairwiseDistance as pdist
def make_node_ref(nodes: torch.Tensor):
    '''
    Makes a node reference to unite node indicies across explanations. 
        Returns a dictionary keyed on node indices in tensor provided.
    Args:
        nodes (torch.tensor): Tensor of nodes to reference.
    
    :rtype: :obj:`Dict`
    '''
    node_reference = {nodes[i].item():i for i in range(nodes.shape[0])}
    return node_reference

def node_mask_from_edge_mask(node_subset: torch.Tensor, edge_index: torch.Tensor, edge_mask: torch.Tensor = None):
    '''
    Gets node mask from an edge_mask:

    Args:
        node_subset (torch.Tensor): Subset of nodes to include in the mask (i.e. that become True).
        edge_index (torch.Tensor): Full edge index of graph.
        edge_mask (torch.Tensor): Boolean mask over all edges in edge_index. Shape: (edge_index.shape[1],).
    '''
    if edge_mask is not None:
        mask_eidx = edge_index[:,edge_mask]
    else:
        mask_eidx = edge_index

    unique_nodes = torch.unique(mask_eidx)

    node_mask = torch.tensor([node_subset[i] in unique_nodes for i in range(node_subset.shape[0])])
    
    return node_mask.float()

def edge_mask_from_node_mask(node_mask: torch.Tensor, edge_index: torch.Tensor):
    '''
    Convert edge_mask to node_mask

    Args:
        node_mask (torch.Tensor): Boolean mask over all nodes included in edge_index. Indices must 
            match to those in edge index. This is straightforward for graph-level prediction, but 
            converting over subgraphs must be done carefully to match indices in both edge_index and
            the node_mask.
    '''

    node_numbers = node_mask.nonzero(as_tuple=True)[0]

    iter_mask = torch.zeros((edge_index.shape[1],))

    # See if edges have both ends in the node mask
    for i in range(edge_index.shape[1]):
        iter_mask[i] = (edge_index[0,i] in node_numbers) and (edge_index[1,i] in node_numbers) 
    
    return iter_mask


def top_k_mask(to_mask: torch.Tensor, top_k: int):
    '''
    Perform a top-k mask on to_mask tensor.

    ..note:: Deals with identical values in the same way as
        torch.sort.

    Args:
        to_mask (torch.Tensor): Tensor to mask.
        top_k (int): How many features in Tensor to select.

    :rtype: :obj:`torch.Tensor`
    Returns:
        torch.Tensor: Masked version of to_mask
    '''
    inds = torch.argsort(to_mask)[-int(top_k):]
    mask = torch.zeros_like(to_mask)
    mask[inds] = 1
    return mask.long()

def threshold_mask(to_mask: torch.Tensor, threshold: float):
    '''
    Perform a threshold mask on to_mask tensor.

    Args:
        to_mask (torch.Tensor): Tensor to mask.
        threshold (float): Select all values greater than this threshold.

    :rtype: :obj:`torch.Tensor`
    Returns:
        torch.Tensor: Masked version of to_mask.
    '''
    return (to_mask > threshold).long()

def distance(emb_1: torch.tensor, emb_2: torch.tensor, p=2) -> float:
    '''
    Calculates the distance between embeddings generated by a GNN model
    Args:
        emb_1 (torch.tensor): embeddings for the clean graph
        emb_2 (torch.tensor): embeddings for the perturbed graph
    '''
    if p == 0:
        return torch.dist(emb_1, emb_2, p=0).item()
    elif p == 1:
        return torch.dist(emb_1, emb_2, p=1).item()
    elif p == 2:
        return torch.dist(emb_1, emb_2, p=2).item()
    else:
        print('Invalid choice! Exiting..')

def match_edge_presence(edge_index, node_idx):
    '''
    Returns edge mask with the spots containing node_idx highlighted
    '''

    emask = torch.zeros(edge_index.shape[1]).bool()

    if isinstance(node_idx, torch.Tensor):
        if node_idx.shape[0] > 1:
            for ni in node_idx:
                emask = emask | ((edge_index[0,:] == ni) | (edge_index[1,:] == ni))
        else:
            emask = ((edge_index[0,:] == node_idx) | (edge_index[1,:] == node_idx))
    else:
        emask = ((edge_index[0,:] == node_idx) | (edge_index[1,:] == node_idx))

    return emask

class EnclosingSubgraph:
    '''
    Args: 
        nodes (torch.LongTensor): Nodes in subgraph.
        edge_index (torch.LongTensor): Edge index for subgraph 
        inv (torch.LongTensor): Inversion of nodes in subgraph (see
            torch_geometric.utils.k_hop_subgraph method.)
        edge_mask (torch.BoolTensor): Mask of edges in entire graph.
        directed (bool, optional): If True, subgraph is directed. 
            (:default: :obj:`False`)
    '''
    def __init__(self, 
            nodes: torch.LongTensor, 
            edge_index: torch.LongTensor, 
            inv: torch.LongTensor, 
            edge_mask: torch.BoolTensor, 
            directed: Optional[bool] = False
        ):

        self.nodes = nodes
        self.edge_index = edge_index
        self.inv = inv
        self.edge_mask = edge_mask
        self.directed = directed

    def draw(self, show = False):
        G = to_networkx_conv(Data(edge_index=self.edge_index), to_undirected=True)
        nx.draw(G)
        if show:
            plt.show()
            
class Explanation:
    '''
    Members:
        feature_imp (torch.Tensor): Feature importance scores
            - Size: (x1,) with x1 = number of features
        node_imp (torch.Tensor): Node importance scores
            - Size: (n,) with n = number of nodes in subgraph or graph
        edge_imp (torch.Tensor): Edge importance scores
            - Size: (e,) with e = number of edges in subgraph or graph
        node_idx (int): Index for node explained by this instance
        node_reference (tensor of ints): Tensor matching length of `node_reference` 
            which maps each index onto original node in the graph
        edge_reference (tensor of ints): Tensor maching lenght of `edge_reference`
            which maps each index onto original edge in the graph's edge
            index
        graph (torch_geometric.data.Data): Original graph on which explanation
            was computed
            - Optional member, can be left None if graph is too large
    Optional members:
        enc_subgraph (Subgraph): k-hop subgraph around 
            - Corresponds to nodes and edges comprising computational graph around node
    '''
    def __init__(self,
        feature_imp: Optional[torch.tensor] = None,
        node_imp: Optional[torch.tensor] = None,
        edge_imp: Optional[torch.tensor] = None,
        node_idx: Optional[torch.tensor] = None,
        node_reference: Optional[torch.tensor] = None,
        edge_reference: Optional[torch.tensor] = None,
        graph = None):

        # Establish basic properties
        self.feature_imp = feature_imp
        self.node_imp = node_imp
        self.edge_imp = edge_imp

        # Only established if passed explicitly in init, not overwritten by enclosing subgraph 
        #   unless explicitly specified
        self.node_reference = node_reference
        self.edge_reference = edge_reference

        self.node_idx = node_idx # Set this for node-level prediction explanations
        self.graph = graph

    def set_enclosing_subgraph(self, subgraph):
        '''
        Args:
            subgraph (tuple, EnclosingSubgraph, or nx.Graph): Return value from torch_geometric.utils.k_hop_subgraph
        '''
        if isinstance(subgraph, EnclosingSubgraph):
            self.enc_subgraph = subgraph
        elif isinstance(subgraph, nx.Graph):
            # Convert from nx.Graph
            data = from_networkx(subgraph)
            nodes = torch.unique(data.edge_index)
            # TODO: Support inv and edge_mask through networkx
            self.enc_subgraph = EnclosingSubgraph(
                nodes = nodes,
                edge_index = data.edge_index,
                inv = None,
                edge_mask = None
            )
        else: # Assumed to be a tuple:
            self.enc_subgraph = EnclosingSubgraph(*subgraph)

        if self.node_reference is None:
            self.node_reference = make_node_ref(self.enc_subgraph.nodes)

    def apply_subgraph_mask(self, 
        mask_node: Optional[bool] = False, 
        mask_edge: Optional[bool] = False):
        '''
        Performs automatic masking on the node and edge importance members

        Args:
            mask_node (bool, optional): If True, performs masking on node_imp based on enclosing subgraph nodes.
                Assumes that node_imp is set for entire graph and then applies mask.
            mask_edge (bool, optional): If True, masks edges in edge_imp based on enclosing subgraph edge mask.

        Example workflow:
        >>> exp = Explanation()
        >>> exp.node_imp = node_importance_tensor
        >>> exp.edge_imp = edge_importance_tensor
        >>> exp.set_enclosing_subgraph(k_hop_subgraph(node_idx, k, edge_index))
        >>> exp.apply_subgraph_mask(True, True) # Masks both node and edge importance
        '''
        if mask_edge:
            mask_inds = self.enc_subgraph.edge_mask.nonzero(as_tuple=True)[0]
            self.edge_imp = self.edge_imp[mask_inds] # Perform masking on current member
        if mask_node:
            self.node_imp = self.node_imp[self.enc_subgraph.nodes]

    def set_whole_graph(self, data: Data):
        '''
        Args:
            data (torch_geometric.data.Data): Data object representing the graph to store.
        
        :rtype: :obj:`None`
        '''
        self.graph = data

    def graph_to_networkx(self, 
        to_undirected=False, 
        remove_self_loops: Optional[bool]=False,
        get_map: Optional[bool] = False):
        '''
        Convert graph to Networkx Graph

        Args:
            to_undirected (bool, optional): If True, graph is undirected. (:default: :obj:`False`)
            remove_self_loops (bool, optional): If True, removes all self-loops in graph.
                (:default: :obj:`False`)
            get_map (bool, optional): If True, returns a map of nodes in graph 
                to nodes in the Networkx graph. (:default: :obj:`False`)

        :rtype: :class:`Networkx.Graph` or :class:`Networkx.DiGraph`
            If `get_map == True`, returns tuple: (:class:`Networkx.Graph`, :class:`dict`)
        '''

        if to_undirected:
            G = nx.Graph()
        else:
            G = nx.DiGraph()

        node_list = sorted(torch.unique(self.graph.edge_index).tolist())
        map_norm =  {node_list[i]:i for i in range(len(node_list))}

        G.add_nodes_from([map_norm[n] for n in node_list])

        # Assign values to each node:
        # Skipping for now

        for i, (u, v) in enumerate(self.graph.edge_index.t().tolist()):
            u = map_norm[u]
            v = map_norm[v]

            if to_undirected and v > u:
                continue

            if remove_self_loops and u == v:
                continue

            G.add_edge(u, v)

            # No edge_attr additions added now
            if self.edge_imp is not None:
                G[u][v]['edge_imp'] = self.edge_imp[i].item()
            # for key in edge_attrs if edge_attrs is not None else []:
            #     G[u][v][key] = values[key][i]

        if self.node_imp is not None:
            for i, feat_dict in G.nodes(data=True):
                # self.node_imp[i] should be a scalar value
                feat_dict.update({'node_imp': self.node_imp[map_norm[i]].item()})

        if get_map:
            return G, map_norm

        return G

    def enc_subgraph_to_networkx(self, 
        to_undirected=False, 
        remove_self_loops: Optional[bool]=False,
        get_map: Optional[bool] = False):
        '''
        Convert enclosing subgraph to Networkx Graph

        Args:
            to_undirected (bool, optional): If True, graph is undirected. (:default: :obj:`False`)
            remove_self_loops (bool, optional): If True, removes all self-loops in graph.
                (:default: :obj:`False`)
            get_map (bool, optional): If True, returns a map of nodes in enclosing subgraph 
                to nodes in the Networkx graph. (:default: :obj:`False`)

        :rtype: :class:`Networkx.Graph` or :class:`Networkx.DiGraph`
            If `get_map == True`, returns tuple: (:class:`Networkx.Graph`, :class:`dict`)
        '''

        if to_undirected:
            G = nx.Graph()
        else:
            G = nx.DiGraph()

        node_list = sorted(torch.unique(self.enc_subgraph.edge_index).tolist())
        map_norm =  {node_list[i]:i for i in range(len(node_list))}
        rev_map = {v:k for k,v in map_norm.items()}

        G.add_nodes_from([map_norm[n] for n in node_list])

        # Assign values to each node:
        # Skipping for now

        for i, (u, v) in enumerate(self.enc_subgraph.edge_index.t().tolist()):
            u = map_norm[u]
            v = map_norm[v]

            if to_undirected and v > u:
                continue

            if remove_self_loops and u == v:
                continue

            G.add_edge(u, v)

            if self.edge_imp is not None:
                G.edges[u, v]['edge_imp'] = self.edge_imp[i].item()

        if self.node_imp is not None:
            for i, feat_dict in G.nodes(data=True):
                # self.node_imp[i] should be a scalar value
                feat_dict.update({'node_imp': self.node_imp[i].item()})

        if get_map:
            return G, map_norm

        return G

    def top_k_node_imp(self, top_k: int, inplace = False):
        '''
        Top-k masking of the node importance for this Explanation.

        Args:
            top_k (int): How many highest scores to include in the mask.
            inplace (bool, optional): If True, masks the node_imp member 
                of the class.

        :rtype: :obj:`torch.Tensor`
        '''

        if inplace:
            self.node_imp = top_k_mask(self.node_imp, top_k)
        else:
            return top_k_mask(self.node_imp, top_k)

    def top_k_edge_imp(self, top_k: int, inplace = False):
        '''
        Top-k masking of the edge importance for this Explanation.

        Args:
            top_k (int): How many highest scores to include in the mask.
            inplace (bool, optional): If True, masks the node_imp member 
                of the class.

        :rtype: :obj:`torch.Tensor`
        '''
        if inplace:
            self.edge_imp = top_k_mask(self.edge_imp, top_k)
        else:
            return top_k_mask(self.edge_imp, top_k)

    def top_k_feature_imp(self, top_k: int, inplace = False):
        '''
        Top-k masking of the feature importance for this Explanation.

        Args:
            top_k (int): How many highest scores to include in the mask.
            inplace (bool, optional): If True, masks the node_imp member 
                of the class.

        :rtype: :obj:`torch.Tensor`
        '''
        if inplace:
            self.feature_imp = top_k_mask(self.feature_imp, top_k)
        else:
            return top_k_mask(self.feature_imp, top_k)

    def thresh_node_imp(self, threshold: float, inplace = False):
        '''
        Threshold mask the node importance

        Args:
            threshold (float): Select all values greater than this value.
            inplace (bool, optional): If True, masks the node_imp member 
                of the class.

        :rtype: :obj:`torch.Tensor`
        '''
        if inplace:
            self.node_imp = threshold_mask(self.node_imp, threshold)
        else:
            return threshold_mask(self.node_imp, threshold)

    def thresh_edge_imp(self, threshold: float, inplace = False):
        '''
        Threshold mask the edge importance

        Args:
            threshold (float): Select all values greater than this value.
            inplace (bool, optional): If True, masks the node_imp member 
                of the class.

        :rtype: :obj:`torch.Tensor`
        '''
        if inplace:
            self.edge_imp = threshold_mask(self.edge_imp, threshold)
        else:
            return threshold_mask(self.edge_imp, threshold)

    def thresh_feature_imp(self, threshold: float, inplace = False):
        '''
        Threshold mask the feature importance

        Args:
            threshold (float): Select all values greater than this value.
            inplace (bool, optional): If True, masks the node_imp member 
                of the class.

        :rtype: :obj:`torch.Tensor`
        '''
        if inplace:
            self.feature_imp = threshold_mask(self.feature_imp, threshold)
        else:
            return threshold_mask(self.feature_imp, threshold)

    def visualize_node(self, 
            num_hops: int,
            graph_data: Data = None,
            additional_hops: int = 1, 
            heat_by_prescence: bool = False, 
            heat_by_exp: bool = True, 
            node_agg_method: str = 'sum',
            ax: matplotlib.axes.Axes = None,
            show: bool = False,
            show_node_labels: bool = False,
            norm_imps = False,
        ):
        '''
        Shows the explanation in context of a few more hops out than its k-hop neighborhood. Used for
            visualizing the explanation for a node-level prediction task.
        
        ..note:: If neither `heat_by_prescence` or `heat_by_exp` are true, the method plots a simple
            visualization of the subgraph around the focal node.

        Args:
            num_hops (int): Number of hops in the enclosing subgraph.
            graph_data (torch_geometric.data.Data, optional): Data object containing graph. Don't provide
                if already stored in the dataset. Used so large graphs can be stored externally and used
                for visualization. (:default: :obj:`None`)
            additional_hops (int, optional): Additional number of hops to include for the visualization.
                If the size of the enclosing subgraph for a node `v` with respect to some model `f` 
                is `n`, then we would show the `n + additional_hops`-hop neighborhood around `v`.
                (:default: :obj:`1`)
            heat_by_prescence (bool, optional): If True, only highlights nodes in the enclosing subgraph.
                Useful for debugging or non-explanation visualization. (:default: :obj:`False`)
            heat_by_exp (bool, optional): If True, highlights nodes and edges by explanation values. 
                (:default: :obj:`True`)
            node_agg_method (str, optional): Aggregation method to use for showing multi-dimensional
                node importance scores (i.e. across features, such as GuidedBP or Vanilla Gradient).
                Options: :obj:`'sum'` and :obj:`'max'`. (:default: :obj:`'sum'`)
            ax (matplotlib.axes.Axes, optional): Axis on which to draw. If not provided, draws directly
                to plt. (:default: :obj:`None`)
            show (bool, optional): If True, shows the plot immediately after drawing. (:default: :obj:`False`)
            show_node_labels (bool, optional): If True, shows the node labels as integers overlaid on the 
                plot. (:default: :obj:`False`)
        '''

        assert self.node_idx is not None, "visualize_node only for node-level explanations, but node_idx is None" 

        #data_G = self.graph.get_Data()
        wholeG = to_networkx_conv(graph_data, to_undirected=True)
        kadd_hop_neighborhood = khop_subgraph_nx(
                G = wholeG, 
                num_hops= num_hops + additional_hops, 
                node_idx=self.node_idx
            )

        subG = wholeG.subgraph(kadd_hop_neighborhood)

        node_agg = torch.sum if node_agg_method == 'sum' else torch.max

        # Identify highlighting nodes:
        exp_nodes = self.enc_subgraph.nodes

        draw_args = dict()

        if norm_imps:
            # Normalize all importance scores:
            save_imps = [self.node_imp, self.edge_imp, self.feature_imp]
            save_imps = [s.clone() if s is not None else s for s in save_imps ]
            for s in (self.node_imp, self.edge_imp, self.feature_imp):
                if s is not None:
                    s = s / s.sum()

        if heat_by_prescence:
            if self.node_imp is not None:
                node_c = [int(i in exp_nodes) for i in subG.nodes]
                draw_args['node_color'] = node_c

        if heat_by_exp:
            if self.node_imp is not None:
                node_c = []
                for i in subG.nodes:
                    if i in self.enc_subgraph.nodes:
                        if isinstance(self.node_imp[self.node_reference[i]], torch.Tensor):
                            if self.node_imp[self.node_reference[i]].dim() > 0:
                                c = node_agg(self.node_imp[self.node_reference[i]]).item()
                            else:
                                c = self.node_imp[self.node_reference[i]].item()
                        else:
                            c = self.node_imp[self.node_reference[i]]
                    else:
                        c = 0

                    node_c.append(c)

                draw_args['node_color'] = node_c

            if self.edge_imp is not None:
                whole_edge_index, _ = subgraph(kadd_hop_neighborhood, edge_index = graph_data.edge_index)

                # Need to match edge indices across edge_index and edges in graph
                tuple_edge_index = [(whole_edge_index[0,i].item(), whole_edge_index[1,i].item()) \
                    for i in range(whole_edge_index.shape[1])]

                _, emask = remove_duplicate_edges(self.enc_subgraph.edge_index)
                # Remove self loops:
                emask_2 = torch.logical_not(self.enc_subgraph.edge_index[0,:] == self.enc_subgraph.edge_index[1,:])
                emask = emask & emask_2

                trimmed_enc_subg_edge_index = self.enc_subgraph.edge_index[:,emask]

                mask_edge_imp = self.edge_imp[emask].clone()

                # Find where edge_imp is applied on one duplicate edge but not another:
                masked_out_by_rmdup = self.enc_subgraph.edge_index[:,torch.logical_not(emask)]
                ones_in_rmdup = self.edge_imp[torch.logical_not(emask)].nonzero(as_tuple=True)[0]
                for j in ones_in_rmdup:
                    edge = masked_out_by_rmdup[:,j].tolist()
                    # Reverse the edge:
                    edge = edge[::-1] 
                    
                    trim_loc_mask = (trimmed_enc_subg_edge_index[0,:] == edge[0]) & (trimmed_enc_subg_edge_index[1,:] == edge[1])
                    trim_loc = (trim_loc_mask).nonzero(as_tuple=True)[0] 
                    if trim_loc.shape[0] > 0:
                        # Should be over 0 if we found it
                        trim_loc = trim_loc[0].item()
                        mask_edge_imp[trim_loc] = 1 # Ensure this edge is also one
                    # Don't do anything if we didn't find it

                positive_edge_indices = mask_edge_imp.nonzero(as_tuple=True)[0]

                # TODO: fix edge imp vis. to handle continuous edge importance scores
                mask_edge_imp = self.edge_imp[positive_edge_indices]

                positive_edges = [(trimmed_enc_subg_edge_index[0,e].item(), trimmed_enc_subg_edge_index[1,e].item()) \
                    for e in positive_edge_indices]

                # Tuples in list should be hashable
                edge_list = list(subG.edges)

                # Get dictionary with mapping from edge index to networkx graph
                #edge_matcher = match_torch_to_nx_edges(subG, remove_duplicate_edges(whole_edge_index)[0])
                edge_matcher = {edge_list[i]:i for i in range(len(edge_list))}
                for i in range(len(edge_list)):
                    forward_tup = edge_list[i]
                    backward_tup = tuple(list(edge_list[i])[::-1])
                    edge_matcher[forward_tup] = i
                    edge_matcher[backward_tup] = i

                edge_heat = torch.zeros(len(edge_list))
                #edge_heat = torch.zeros(whole_edge_index.shape[1])

                for e in positive_edges:
                    #e = positive_edges[i]
                    # Must find index, which is not very efficient
                    edge_heat[edge_matcher[e]] = 1

                draw_args['edge_color'] = edge_heat.tolist()
                #coolwarm cmap:
                draw_args['edge_cmap'] = plt.cm.coolwarm

            # Heat edge explanations if given

        # Seed the position to stay consistent:
        pos = nx.spring_layout(subG, seed = 1234)
        nx.draw(subG, pos, ax = ax, **draw_args, with_labels = show_node_labels)

        # Highlight the center node index:
        nx.draw(subG.subgraph(self.node_idx), pos, node_color = 'red', 
                node_size = 400, ax = ax)

        if norm_imps:
            self.node_imp, self.edge_imp, self.feature_imp = save_imps[0], save_imps[1], save_imps[2]

        if show:
            plt.show()

def get_flag():
    pass

# Set common shapes:
house = nx.house_graph()
house_x = nx.house_x_graph()
diamond = nx.diamond_graph()
pentagon = nx.cycle_graph(n=5)
wheel = nx.wheel_graph(n=6)
star = nx.star_graph(n=5)
flag = None

triangle = nx.Graph()
triangle.add_nodes_from([0, 1, 2])
triangle.add_edges_from([(0, 1), (1, 2), (2, 0)])

def random_shape(n) -> nx.Graph:
    '''
    Outputs a random shape as nx.Graph

    ..note:: set `random.seed()` for seeding
    
    Args:
        n (int): Number of shapes in the bank to draw from
    
    '''
    shape_list = [
        house,
        pentagon,
        wheel
    ]
    i = random.choice(list(range(len(shape_list))))
    return shape_list[i], i + 1

from sklearn.model_selection import train_test_split

class NodeDataset:
    def __init__(self, 
        name, 
        num_hops: int,
        download: Optional[bool] = False,
        root: Optional[str] = None
        ):
        self.name = name
        self.num_hops = num_hops
    def get_graph(self, 
        use_fixed_split: bool = True, 
        split_sizes: Tuple = (0.7, 0.2, 0.1),
        stratify: bool = True, 
        seed: int = None):
        if sum(split_sizes) != 1: # Normalize split sizes
            split_sizes = np.array(split_sizes) / sum(split_sizes)
        if use_fixed_split:
            self.graph.train_mask = self.fixed_train_mask
            self.graph.valid_mask = self.fixed_valid_mask
            self.graph.test_mask  = self.fixed_test_mask
        else:
            assert len(split_sizes) == 3, "split_sizes must contain (train_size, test_size, valid_size)"
            # Create a split for user (based on seed, etc.)
            train_mask, test_mask = train_test_split(list(range(self.graph.num_nodes)), 
                                test_size = split_sizes[1] + split_sizes[2], 
                                random_state = seed, stratify = self.graph.y.tolist() if stratify else None)
            if split_sizes[2] > 0:
                valid_mask, test_mask = train_test_split(test_mask, 
                                    test_size = split_sizes[2] / split_sizes[1],
                                    random_state = seed, stratify = self.graph.y[test_mask].tolist() if stratify else None)
                self.graph.valid_mask = torch.tensor([i in valid_mask for i in range(self.graph.num_nodes)], dtype = torch.bool)
            self.graph.train_mask = torch.tensor([i in train_mask for i in range(self.graph.num_nodes)], dtype = torch.bool)
            self.graph.test_mask  = torch.tensor([i in test_mask  for i in range(self.graph.num_nodes)], dtype = torch.bool)
        return self.graph
    def download(self):
        '''TODO: Implement'''
        pass
    def get_enclosing_subgraph(self, node_idx: int):
        '''
        Args:
            node_idx (int): Node index for which to get subgraph around
        '''
        k_hop_tuple = k_hop_subgraph(node_idx, 
            num_hops = self.num_hops, 
            edge_index = self.graph.edge_index)
        return EnclosingSubgraph(*k_hop_tuple)
    def nodes_with_label(self, label = 0, mask = None) -> torch.Tensor:
        '''
        Get all nodes that are a certain label
        Args:
            label (int, optional): Label for which to find nodes.
                (:default: :obj:`0`)

        Returns:
            torch.Tensor: Indices of nodes that are of the label
        '''
        if mask is not None:
            return ((self.graph.y == label) & (mask)).nonzero(as_tuple=True)[0]
        return (self.graph.y == label).nonzero(as_tuple=True)[0]

    def choose_node_with_label(self, label = 0, mask = None):
        '''
        Choose a random node with a given label
        Args:
            label (int, optional): Label for which to find node.
                (:default: :obj:`0`)

        Returns:
            tuple(int, Explanation):
                int: Node index found
                Explanation: explanation corresponding to that node index
        '''
        nodes = self.nodes_with_label(label = label, mask = mask)
        node_idx = random.choice(nodes).item()
        return node_idx, self.explanations[node_idx]

    def nodes_in_shape(self, inshape = True, mask = None):
        '''
        Get a group of nodes by shape membership.

        Args:
            inshape (bool, optional): If the nodes are in a shape.
                :obj:`True` means that the nodes returned are in a shape.
                :obj:`False` means that the nodes are not in a shape.

        Returns:
            torch.Tensor: All node indices for nodes in or not in a shape.
        '''
        # Get all nodes in a shape
        condition = (lambda n: self.G.nodes[n]['shape'] > 0) if inshape \
                else (lambda n: self.G.nodes[n]['shape'] == 0)
        if mask is not None:
            condition = (lambda n: (condition(n) and mask[n].item()))
        return torch.tensor([n for n in self.G.nodes if condition(n)]).long()

    def choose_node_in_shape(self, inshape = True, mask = None):
        '''
        Gets a random node by shape membership.

        Args:
            inshape (bool, optional): If the node is in a shape.
                :obj:`True` means that the node returned is in a shape.
                :obj:`False` means that the node is not in a shape.

        Returns:
            Tuple[int, Explanation]
                int: Node index found
                Explanation: Explanation corresponding to that node index
        '''
        nodes = self.nodes_in_shape(inshape = inshape, mask = mask)
        node_idx = random.choice(nodes).item()
        return node_idx, self.explanations[node_idx]


    def choose_node(self, inshape = None, label = None, split = None):
        '''
        Chooses random nodes in the graph. Has support for multiple logical
            indexing.

        Args:
            inshape (bool, optional): If the node is in a shape.
                :obj:`True` means that the node returned is in a shape.
                :obj:`False` means that the node is not in a shape.
            label (int, optional): Label for which to find node.
                (:default: :obj:`0`)
        
        Returns:
        '''
        split = split.lower() if split is not None else None

        if split == 'validation' or split == 'valid' or split == 'val':
            split = 'val'

        map_to_mask = {
            'train': self.graph.train_mask,
            'val': self.graph.valid_mask,
            'test': self.graph.test_mask,
        }
        
        # Get mask based on provided string:
        mask = None if split is None else map_to_mask[split]

        if inshape is None:
            if label is None:
                to_choose = torch.arange(end = self.num_nodes)
            else:
                to_choose = self.nodes_with_label(label = label, mask = mask)
        
        elif label is None:
            to_choose = self.nodes_in_shape(inshape = inshape, mask = mask)

        else:
            t_inshape = self.nodes_in_shape(inshape = inshape, mask = mask)
            t_label = self.nodes_with_label(label = label, make = mask)

            # Joint masking over shapes and labels:
            to_choose = torch.as_tensor([n.item() for n in t_label if n in t_inshape]).long()

        assert_fmt = 'Could not find a node in {} with inshape={}, label={}'
        assert to_choose.nelement() > 0, assert_fmt.format(self.name, inshape, label)

        node_idx = random.choice(to_choose).item()
        return node_idx, self.explanations[node_idx]

    def __len__(self) -> int:
        return 1 # There is always just one graph

    def dump(self, fname = None):
        fname = self.name + '.pickle' if fname is None else fname
        torch.save(self, open(fname, 'wb'))
    @property
    def x(self):
        return self.graph.x

    @property
    def edge_index(self):
        return self.graph.edge_index

    def y(self):
        return self.graph.y

    def __getitem__(self, idx):
        assert idx == 0, 'Dataset has only one graph'
        return self.graph, self.explanation

class GraphDataset:
    def __init__(self, name, split_sizes = (0.7, 0.2, 0.1), seed = None, device = None):
        self.name = name
        self.seed = seed
        self.device = device
        if split_sizes[1] > 0:
            self.train_index, self.test_index = train_test_split(torch.arange(start = 0, end = len(self.graphs)), 
                test_size = split_sizes[1] + split_sizes[2], random_state=self.seed, shuffle = False)
        else:
            self.test_index = None
            self.train_index = torch.arange(start = 0, end = len(self.graphs))
        if split_sizes[2] > 0:
            self.test_index, self.val_index = train_test_split(self.test_index, 
                test_size = split_sizes[2] / (split_sizes[1] + split_sizes[2]),
                random_state = self.seed, shuffle = False)
        else:
            self.val_index = None
        self.Y = torch.tensor([self.graphs[i].y for i in range(len(self.graphs))]).to(self.device)
    def get_data_list(
            self,
            index,
        ):
        data_list = [self.graphs[i].to(self.device) for i in index]
        exp_list = [self.explanations[i] for i in index]
        return data_list, exp_list
    def get_loader(
            self, 
            index,
            batch_size = 16,
            **kwargs
        ):
        data_list, exp_list = self.get_data_list(index)
        for i in range(len(data_list)):
            data_list[i].exp_key = [i]
        loader = DataLoader(data_list, batch_size = batch_size, shuffle = True)
        return loader, exp_list
    def get_train_loader(self, batch_size = 16):
        return self.get_loader(index=self.train_index, batch_size = batch_size)
    def get_train_list(self):
        return self.get_data_list(index = self.train_index)
    def get_test_loader(self):
        assert self.test_index is not None, 'test_index is None'
        return self.get_loader(index=self.test_index, batch_size = 1)
    def get_test_list(self):
        assert self.test_index is not None, 'test_index is None'
        return self.get_data_list(index = self.test_index)
    def get_val_loader(self):
        assert self.test_index is not None, 'val_index is None'
        return self.get_loader(index=self.val_index, batch_size = 1)
    def get_val_list(self):
        assert self.val_index is not None, 'val_index is None'
        return self.get_data_list(index = self.val_index)
    def get_train_w_label(self, label):
        inds_to_choose = (self.Y[self.train_index] == label).nonzero(as_tuple=True)[0]
        in_train_idx = inds_to_choose[torch.randint(low = 0, high = inds_to_choose.shape[0], size = (1,))]
        chosen = self.train_index[in_train_idx.item()]
        return self.graphs[chosen], self.explanations[chosen]
    def get_test_w_label(self, label):
        assert self.test_index is not None, 'test_index is None'
        inds_to_choose = (self.Y[self.test_index] == label).nonzero(as_tuple=True)[0]
        in_test_idx = inds_to_choose[torch.randint(low = 0, high = inds_to_choose.shape[0], size = (1,))]
        chosen = self.test_index[in_test_idx.item()]
        return self.graphs[chosen], self.explanations[chosen]
    def get_graph_as_networkx(self, graph_idx):
        '''
        Get a given graph as networkx graph
        '''
        g = self.graphs[graph_idx]
        return to_networkx_conv(g, node_attrs = ['x'], to_undirected=True)
    def download(self):
        pass
    def __getitem__(self, idx):
        return self.graphs[idx], self.explanations[idx]
    def __len__(self):
        return len(self.graphs)

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)
                     
def _generate_hypercube(samples, dimensions, rng):
    """
    Returns distinct binary samples of length dimensions.
    """
    if dimensions > 30:
        return np.hstack([rng.randint(2, size=(samples, dimensions - 30)),
                          _generate_hypercube(samples, 30, rng)])
    out = sample_without_replacement(2 ** dimensions, samples,
                                     random_state=rng).astype(dtype='>u4', copy=False)
    out = np.unpackbits(out.view('>u1')).reshape((-1, 32))[:, -dimensions:]
    return out


def make_structured_feature(y: torch.Tensor, n_features=5, n_informative=2,
                            n_redundant=0, n_repeated=0, n_clusters_per_class=2,
                            unique_explanation=True, flip_y=0.01,
                            class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
                            shuffle=True, seed=None):
    """This function is based on sklearn.datasets.make_classification.

    Generate structured features for the given labels.

    This initially creates clusters of points normally distributed (std=1)
    about vertices of an ``n_informative``-dimensional hypercube with sides of
    length ``2*class_sep`` and assigns an equal number of clusters to each
    class. It introduces interdependence between these features and adds
    various types of further noise to the data.

    Without shuffling, ``X`` horizontally stacks features in the following
    order: the primary ``n_informative`` features, followed by ``n_redundant``
    linear combinations of the informative features, followed by ``n_repeated``
    duplicates, drawn randomly with replacement from the informative and
    redundant features. The remaining features are filled with random noise.
    Thus, without shuffling, all useful features are contained in the columns
    ``X[:, :n_informative + n_redundant + n_repeated]``.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        The integer labels for class membership of each sample.

    n_samples : int, default=100
        The number of samples.

    n_features : int, default=20
        The total number of features. These comprise ``n_informative``
        informative features, ``n_redundant`` redundant features,
        ``n_repeated`` duplicated features and
        ``n_features-n_informative-n_redundant-n_repeated`` useless features
        drawn at random.

    n_informative : int, default=2
        The number of informative features. Each class is composed of a number
        of gaussian clusters each located around the vertices of a hypercube
        in a subspace of dimension ``n_informative``. For each cluster,
        informative features are drawn independently from  N(0, 1) and then
        randomly linearly combined within each cluster in order to add
        covariance. The clusters are then placed on the vertices of the
        hypercube.

    n_redundant : int, default=2
        The number of redundant features. These features are generated as
        random linear combinations of the informative features.

    n_repeated : int, default=0
        The number of duplicated features, drawn randomly from the informative
        and the redundant features.

    n_clusters_per_class : int, default=2
        The number of clusters per class.

    flip_y : float, default=0.01
        The fraction of samples whose class is assigned randomly. Larger
        values introduce noise in the labels and make the classification
        task harder. Note that the default setting flip_y > 0 might lead
        to less than ``n_classes`` in y in some cases.

    class_sep : float, default=1.0
        The factor multiplying the hypercube size.  Larger values spread
        out the clusters/classes and make the classification task easier.

    hypercube : bool, default=True
        If True, the clusters are put on the vertices of a hypercube. If
        False, the clusters are put on the vertices of a random polytope.

    shift : float, ndarray of shape (n_features,) or None, default=0.0
        Shift features by the specified value. If None, then features
        are shifted by a random value drawn in [-class_sep, class_sep].

    scale : float, ndarray of shape (n_features,) or None, default=1.0
        Multiply features by the specified value. If None, then features
        are scaled by a random value drawn in [1, 100]. Note that scaling
        happens after shifting.

    shuffle : bool, default=True
        Shuffle the samples and the features.

    seed : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The generated samples.

    Notes
    -----
    The algorithm is adapted from Guyon [1] and was designed to generate
    the "Madelon" dataset.

    References
    ----------
    .. [1] I. Guyon, "Design of experiments for the NIPS 2003 variable
           selection benchmark", 2003.

    See Also
    --------
    make_blobs : Simplified variant.
    make_multilabel_classification : Unrelated rng for multilabel tasks.
    """

    Yorg = y.clone().numpy()

    if isinstance(y, torch.Tensor):
        y = y.clone().numpy()

    n_samples = y.shape[0]
    labels, n_samples_per_class = np.unique(y, return_counts=True)
    n_classes = len(labels)

    rng = check_random_state(seed)

    # Set n_redundant and n_repeated to 0 if unique explanation
    if unique_explanation:
        n_redundant = n_repeated = 0

    # Count features, clusters and samples
    if n_informative + n_redundant + n_repeated > n_features:
        raise ValueError("Number of informative, redundant and repeated "
                         "features must sum to less than the number of total"
                         " features")
    # Use log2 to avoid overflow errors
    if n_informative < np.log2(n_classes * n_clusters_per_class):
        msg = "n_classes({}) * n_clusters_per_class({}) must be"
        msg += " smaller or equal 2**n_informative({})={}"
        raise ValueError(msg.format(n_classes, n_clusters_per_class,
                                    n_informative, 2**n_informative))

    n_useless = n_features - n_informative - n_redundant - n_repeated
    n_clusters = n_classes * n_clusters_per_class

    # Distribute samples among clusters
    n_samples_per_cluster = [
        int(n_samples_per_class[k % n_classes] / n_clusters_per_class)
        for k in range(n_clusters)
    ]

    for i in range(n_samples - sum(n_samples_per_cluster)):
        n_samples_per_cluster[i % n_clusters] += 1

    # Initialize X
    X = np.zeros((n_samples, n_features))

    # Build the polytope whose vertices become cluster centroids
    centroids = _generate_hypercube(n_clusters, n_informative,
                                    rng).astype(float, copy=False)

    centroids *= 2 * class_sep
    centroids -= class_sep
    if not hypercube:
        centroids *= rng.rand(n_clusters, 1)
        centroids *= rng.rand(1, n_informative)

    # Initially draw informative features from the standard normal
    X[:, :n_informative] = rng.randn(n_samples, n_informative)

    # Create each cluster; a variant of make_blobs
    stop = 0
    for k, centroid in enumerate(centroids):
        start, stop = stop, stop + n_samples_per_cluster[k]
        y[start:stop] = k % n_classes  # assign labels
        X_k = X[start:stop, :n_informative]  # slice a view of the cluster

        A = 2 * rng.rand(n_informative, n_informative) - 1
        X_k[...] = np.dot(X_k, A)  # introduce random covariance

        X_k += centroid  # shift the cluster to a vertex
        #print('k', k)

    # Create redundant features
    if n_redundant > 0:
        B = 2 * rng.rand(n_informative, n_redundant) - 1
        X[:, n_informative:n_informative + n_redundant] = \
            np.dot(X[:, :n_informative], B)

    # Repeat some features
    if n_repeated > 0:
        n = n_informative + n_redundant
        indices = ((n - 1) * rng.rand(n_repeated) + 0.5).astype(np.intp)
        X[:, n:n + n_repeated] = X[:, indices]

    # Fill useless features
    if n_useless > 0:
        X[:, -n_useless:] = rng.randn(n_samples, n_useless)

    # Randomly replace labels
    if flip_y >= 0.0:
        flip_mask = rng.rand(n_samples) < flip_y
        y[flip_mask] = rng.randint(n_classes, size=flip_mask.sum())

    # Randomly shift and scale
    if shift is None:
        shift = (2 * rng.rand(n_features) - 1) * class_sep
    X += shift

    if scale is None:
        scale = 1 + 100 * rng.rand(n_features)
    X *= scale

    # The binary feature mask (1 for informative features) if unique explanation
    if unique_explanation:
        feature_mask = np.zeros(n_features, dtype=bool)
        feature_mask[:n_informative] = True

    #print('y before shuffle', y)

    if shuffle:
        # Randomly permute features
        indices = np.arange(n_features)
        rng.shuffle(indices)
        X[:, :] = X[:, indices]
        #y = np.array([y[i] for i in indices])
        if unique_explanation:
            feature_mask[:] = feature_mask[indices]

    unique_y = np.sort(np.unique(Yorg))

    #print('y', y)

    #print(unique_y)
    #ysort = np.sort(y)

    Xnew = np.zeros_like(X)
    
    for yval in unique_y:
        ingenerated = np.argwhere(y == yval).flatten()
        #print('ingenerated', ingenerated)
        inorg = np.argwhere(Yorg == yval).flatten()
        #print('inorg', inorg)

        for gen, org in zip(ingenerated, inorg): # Move 
            Xnew[org, :] = X[gen, :]

    #print(Xnew[:10, :])

    # Convert to tensor
    Xnew = torch.from_numpy(Xnew).float()
    feature_mask = torch.from_numpy(feature_mask)

    if unique_explanation:
        return Xnew, feature_mask
    else:
        return Xnew
    
def BBG_old(
        shape: Optional[nx.Graph] = house, 
        num_subgraphs: Optional[int] = 5, 
        inter_sg_connections: Optional[int] = 1,
        prob_connection: Optional[float] = 1,
        num_hops: Optional[int] = 2,
        base_graph: Optional[str] = 'ba',
        seed = None,
        **kwargs,
        ) -> nx.Graph:
    '''
    Creates a synthetic graph with one or two motifs within a given neighborhood and
        then labeling nodes based on the number of motifs around them. 
    Can be thought of as building unique explanations for each node, with either one
        or two motifs being the explanation.
    Args:
        shape (nx.Graph, optional): Motif to be inserted.
        num_subgraphs (int, optional): Number of initial subgraphs to create. Roughly
            controls number of nodes in the graph.
        inter_sg_connections (int, optional): How many connections to be made between
            subgraphs. Higher value will create more inter-connected graph. 
        prob_connection (float, optional): Probability of making connection between 
            subgraphs. Can introduce sparsity and stochasticity to graph generation.
        num_hops (int, optional): Number of hops to consider for labeling a node.
        base_graph (str, optional): Base graph algorithm used to generate each subgraph.
            Options are `'ba'` (Barabasi-Albert) (:default: :obj:`'ba'`)
    '''

    np.random.seed(seed)
    random.seed(seed)

    # Create graph:
    if base_graph == 'ba':
        if 'n_ba' in kwargs:
            subgraph_generator = partial(nx.barabasi_albert_graph, n=kwargs['n_ba'], m=1)
        else:
            subgraph_generator = partial(nx.barabasi_albert_graph, n=5 * num_hops, m=1)

    subgraphs = []
    shape_node_per_subgraph = []
    original_shapes = []
    floor_counter = 0
    shape_number = 1
    for i in range(num_subgraphs):
        current_shape = shape.copy()
        #nx.set_node_attributes(current_shape, 1, 'shape')
        #nx.set_node_attributes(current_shape, shape_number, 'shape_number')
        nx.set_node_attributes(current_shape, shape_number, 'shape')

        s = subgraph_generator()
        relabeler = {ns: floor_counter + ns for ns in s.nodes}
        s = nx.relabel.relabel_nodes(s, relabeler)
        nx.set_node_attributes(s, 0, 'shape')
        #nx.set_node_attributes(s, 0, 'shape_number')

        # Join s and shape together:
        to_pivot = random.choice(list(shape.nodes))
        pivot = random.choice(list(s.nodes))

        shape_node_per_subgraph.append(pivot) # This node represents the shape in the graph

        convert = {to_pivot: pivot}

        mx_nodes = max(list(s.nodes))
        i = 1
        for n in current_shape.nodes:
            if not (n == to_pivot):
                convert[n] = mx_nodes + i
            i += 1

        current_shape = nx.relabel.relabel_nodes(current_shape, convert)
        
        s.add_nodes_from(current_shape.nodes(data=True))
        s.add_edges_from(current_shape.edges)

        # Find k-hop from pivot:
        in_house = khop_subgraph_nx(node_idx = pivot, num_hops = num_hops, G = s)
        s.remove_nodes_from(set(s.nodes) - set(in_house) - set(current_shape.nodes))
        nx.set_node_attributes(s, 1, 'shapes_in_khop')

        # Ensure that pivot is assigned to proper shape:
        #s.nodes[pivot]['shape_number'] = shape_number
        s.nodes[pivot]['shape'] = shape_number


        subgraphs.append(s.copy())
        floor_counter = max(list(s.nodes)) + 1
        original_shapes.append(current_shape.copy())

        shape_number += 1

    G = nx.Graph()
    
    for i in range(len(subgraphs)):
        G.add_edges_from(subgraphs[i].edges)
        G.add_nodes_from(subgraphs[i].nodes(data=True))

    G = G.to_undirected()

    # Join subgraphs via inner-subgraph connections
    for i in range(len(subgraphs)):
        for j in range(i + 1, len(subgraphs)):
            #if i == j: # Don't connect the same subgraph
            #    continue

            s = subgraphs[i]
            # Try to make connections between subgraphs i, j:
            for k in range(inter_sg_connections):

                # Screen whether to try to make a connection:
                if np.random.rand() > prob_connection:
                    continue

                x, y = np.meshgrid(list(subgraphs[i].nodes), list(subgraphs[j].nodes))
                possible_edges = list(zip(x.flatten(), y.flatten()))

                rand_edge = None

                tempG = G.copy()

                while len(possible_edges) > 0:

                    rand_edge = random.choice(possible_edges)
                    possible_edges.remove(rand_edge) # Remove b/c we're searching this edge possibility

                    # Make edge between the two:
                    tempG.add_edge(rand_edge[0], rand_edge[1])
                    tempG.add_edge(rand_edge[1], rand_edge[0])
                    #print('rand_edge 1', rand_edge)

                    khop_union = set()

                    # Constant number of t's for each (10)
                    for t in list(original_shapes[i].nodes) + list(original_shapes[j].nodes):
                        khop_union = khop_union.union(set(khop_subgraph_nx(node_idx = t, num_hops = num_hops, G = tempG)))

                    incr_ret = incr_on_unique_houses(
                        nodes_to_search = list(khop_union),   
                        G = tempG, 
                        num_hops = num_hops, 
                        attr_measure = 'shapes_in_khop', 
                        lower_bound = 1, 
                        upper_bound = 2)

                    if incr_ret is None:
                        #print('rand_edge 2', rand_edge)
                        tempG.remove_edge(rand_edge[0], rand_edge[1])
                        #tempG.remove_edge(rand_edge[1], rand_edge[0])

                        rand_edge = None
                        continue
                    else:
                        tempG = incr_ret
                        break

                if rand_edge is not None: # If we found a valid edge
                    #print('Made change')
                    G = tempG.copy()

    # Ensure that G is connected
    G = G.subgraph(sorted(nx.connected_components(G), key = len, reverse = True)[0])


    # Renumber nodes to be constantly increasing integers starting from 0
    mapping = {n:i for i, n in enumerate(G.nodes)}
    G = nx.relabel_nodes(G, mapping = mapping, copy = True)

    return G


def incr_on_unique_houses(nodes_to_search, G, num_hops, attr_measure, lower_bound, upper_bound):
    #G = G.copy()

    incr_tuples = {}

    for n in nodes_to_search:
        khop = khop_subgraph_nx(node_idx = n, num_hops = num_hops, G = G)

        #unique_shapes = torch.unique(torch.tensor([G.nodes[i]['shape_number'] for i in khop]))
        unique_shapes = torch.unique(torch.tensor([G.nodes[i]['shape'] for i in khop]))
        num_unique = unique_shapes.shape[0] - 1 if 0 in unique_shapes else unique_shapes.shape[0]

        if num_unique < lower_bound or num_unique > upper_bound:
            return None
        else:

            incr_tuples[n] = (num_unique, unique_shapes)

            # G.nodes[n][attr_measure] = num_unique
            # G.nodes[n]['nearby_shapes'] = unique_shapes

    for k, v in incr_tuples.items():
        G.nodes[k][attr_measure] = v[0]
        G.nodes[k]['nearby_shapes'] = v[1]

    return G

def ba_around_shape(shape: nx.Graph, add_size: int, show_subgraphs: bool = False):
    '''
    Incrementally adds nodes around a shape in a Barabasi-Albert style

    Args:
        shape (nx.Graph): Shape on which to start the subgraph.
        add_size (int): Additional size of the subgraph, i.e. number of
            nodes to add to the shape to create full subgraph.
        show_subgraphs (bool, optional): If True, shows each subgraph
            through nx.draw. (:default: :obj:`False`)
    '''
    # Get degree, probability distribution of shape

    original_nodes = set(shape.nodes())

    def get_dist():
        degs = [d for n, d in shape.degree() if n in original_nodes]
        total_degree = sum(degs)
        dist = [degs[i]/total_degree for i in range(len(degs))]
        return dist

    node_list = list(shape.nodes())
    top_nodes = max(node_list)

    for i in range(add_size):
        # Must connect to only nodes within original graph
        connect_node = np.random.choice(node_list, p = get_dist())
        new_node = top_nodes + i + 1
        shape.add_node(new_node)
        shape.add_edge(connect_node, new_node) # Just add one edge b/c shape is undirected
        shape.nodes[new_node]['shape'] = 0 # Set to zero because its not in a shape
    
    if show_subgraphs:
        c = [int(not (i in node_list)) for i in shape.nodes]
        nx.draw(shape, node_color = c, cmap = 'brg')
        plt.show()

    return shape

def BBG_PA(
        shape: Optional[nx.Graph] = house, 
        num_subgraphs: Optional[int] = 5, 
        prob_connection: Optional[float] = 1,
        subgraph_size: int = 13,
        seed: int = None,
        **kwargs,
        ) -> nx.Graph:
    '''
    Creates a synthetic graph with one or two motifs within a given neighborhood and
        then labeling nodes based on the number of motifs around them. 
    Can be thought of as building unique explanations for each node, with either one
        or two motifs being the explanation.
    Args:
        shape (nx.Graph, optional): Motif to be inserted.
        num_subgraphs (int, optional): Number of initial subgraphs to create. Roughly
            controls number of nodes in the graph.
        prob_connection (float, optional): Probability of making connection between 
            subgraphs. Can introduce sparsity and stochasticity to graph generation.
        kwargs: Optional arguments
            show_subgraphs (bool): If True, shows each subgraph that is generated during
                initial subgraph generation. (:default: :obj:`False`)
    '''

    subgraphs = []
    original_shapes = []
    floor_counter = 0
    shape_number = 1

    # Option to show individual subgraphs
    show_subgraphs = False if ('show_subgraphs' not in kwargs) or num_subgraphs > 10 else kwargs['show_subgraphs']

    nodes_in_shape = shape.number_of_nodes()

    np.random.seed(seed)
    random.seed(seed)
    #torch.seed(seed)

    for i in range(num_subgraphs):
        current_shape = shape.copy()

        nx.set_node_attributes(current_shape, shape_number, 'shape')

        relabeler = {ns: floor_counter + ns for ns in current_shape.nodes}
        current_shape = nx.relabel.relabel_nodes(current_shape, relabeler)
        original_shapes.append(current_shape.copy())

        subi_size = np.random.poisson(lam = subgraph_size - nodes_in_shape)
        s = ba_around_shape(current_shape, add_size = subi_size, show_subgraphs = show_subgraphs)

        # All nodes have one shape in their k-hop (guaranteed by building procedure)
        nx.set_node_attributes(s, 1, 'shapes_in_khop')

        # Append a copy of subgraph to subgraphs vector
        subgraphs.append(s.copy())

        # Increment floor counter and shape number:
        floor_counter = max(list(s.nodes)) + 1
        shape_number += 1

    G = nx.Graph()
    
    for i in range(len(subgraphs)):
        G.add_edges_from(subgraphs[i].edges)
        G.add_nodes_from(subgraphs[i].nodes(data=True))

    G = G.to_undirected()

    # Make list of possible connections between subgraphs:
    connections = np.array(list(itertools.combinations(np.arange(len(subgraphs)), r = 2)))
    sample_mask = np.random.binomial(n=2, p = prob_connection, size = len(connections)).astype(bool)
    iter_edges = connections[sample_mask]

    # Join subgraphs via inner-subgraph connections
    for i, j in tqdm.tqdm(iter_edges):
        # Try to make connections between subgraphs i, j:

        x, y = np.meshgrid(list(subgraphs[i].nodes), list(subgraphs[j].nodes))
        possible_edges = list(zip(x.flatten(), y.flatten()))

        # Create preferential attachment distribution: -------------------
        deg_dist = np.array([(subgraphs[i].degree(ni) + subgraphs[j].degree(nj)) for ni, nj in possible_edges])
        running_mask = np.ones(deg_dist.shape[0])
        indices_to_choose = np.arange(len(possible_edges))
        # ----------------------------------------------------------------

        rand_edge = None

        #tempG = G.copy()

        while np.sum(running_mask) > 0:

            # -----------
            rand_i = np.random.choice(indices_to_choose, p = deg_dist / np.sum(deg_dist))
            rand_edge = possible_edges[rand_i]
            old_deg = deg_dist[rand_i]
            running_mask[rand_i] = 0

            if np.sum(running_mask) > 0:
                deg_dist = (deg_dist + old_deg/np.sum(running_mask) * running_mask) * running_mask
            # -----------

            # Make edge between the two:
            # tempG.add_edge(rand_edge[0], rand_edge[1])
            # tempG.add_edge(rand_edge[1], rand_edge[0])
            G.add_edge(rand_edge[0], rand_edge[1])
            #print('rand_edge 1', rand_edge)

            khop_union = set()

            # Constant number of t's for each (10)
            for t in list(original_shapes[i].nodes) + list(original_shapes[j].nodes):
                khop_union = khop_union.union(set(khop_subgraph_nx(node_idx = t, num_hops = 1, G = G)))

            incr_ret = incr_on_unique_houses(
                nodes_to_search = list(khop_union),   
                G = G, 
                num_hops = 1, 
                attr_measure = 'shapes_in_khop', 
                lower_bound = 1, 
                upper_bound = 2)

            if incr_ret is None:
                #print('rand_edge 2', rand_edge)
                #empG.remove_edge(rand_edge[0], rand_edge[1])
                G.remove_edge(rand_edge[0], rand_edge[1])
                #tempG.remove_edge(rand_edge[1], rand_edge[0])

                rand_edge = None
                continue
            else:
                #tempG = incr_ret
                G = incr_ret
                break

    # Ensure that G is connected
    G = G.subgraph(sorted(nx.connected_components(G), key = len, reverse = True)[0])

    # Renumber nodes to be constantly increasing integers starting from 0
    mapping = {n:i for i, n in enumerate(G.nodes)}
    G = nx.relabel_nodes(G, mapping = mapping, copy = True)

    return G
    
def verify_motifs(G: nx.Graph, motif_subgraph: nx.Graph):
    '''
    Verifies that all motifs within a graph are "good" motifs
        i.e. they were planted by the building algorithm

    Args:
        G (nx.Graph): Networkx graph on which to search.
        motif_subgraph (nx.Graph): Motif to search for (query graph).

    Returns:
        :rtype: :obj:`bool`
        False if there exists at least one "bad" shape
        True if all motifs/shapes in the graph were planted
    '''

    matcher = nx.algorithms.isomorphism.ISMAGS(graph = G, subgraph = motif_subgraph)

    for iso in matcher.find_isomorphisms():
        nodes_found = iso.keys()
        shapes = [G.nodes[n]['shape'] for n in nodes_found]

        if (sum([int(shapes[i] != shapes[i-1]) for i in range(1, len(shapes))]) > 0) \
            or (sum(shapes) == 0):
            # Found a bad one
            return False

    return True

def if_edge_exists(edge_index: torch.Tensor, node1: int, node2: int):
    '''
    Quick lookup for if an edge exists b/w `node1` and `node2`
    '''
    
    p1 = torch.any((edge_index[0,:] == node1) & (edge_index[1,:] == node2))
    p2 = torch.any((edge_index[1,:] == node1) & (edge_index[0,:] == node2))

    return (p1 or p2).item()

    
def gaussian_lv_generator(
        G: nx.Graph, 
        yvals: torch.Tensor,  
        n_features: int = 10,       
        flip_y: float = 0.01,
        class_sep: float = 1.0,
        n_informative: int = 4,
        n_clusters_per_class: int = 2,
        seed = None):
    '''
    Args:
        G (nx.Graph): 
        yvals (torch.Tensor): 
        seed (seed): (:default: :obj:`None`)
    '''

    x, feature_imp_true = make_structured_feature(
            yvals, 
            n_features = n_features,
            n_informative = n_informative, 
            flip_y = flip_y,
            class_sep=class_sep,
            n_clusters_per_class=n_clusters_per_class,
            seed = seed)

    Gitems = list(G.nodes.items())
    node_map = {Gitems[i][0]:i for i in range(G.number_of_nodes())}

    def get_feature(node_idx):
        return x[node_map[node_idx],:]

    return get_feature, feature_imp_true
    
def motif_id_label(G, num_hops):
    '''
    Gets labels based on motif label in the neighborhood
    '''
    def get_label(node_idx):
        nodes_in_khop = khop_subgraph_nx(node_idx, num_hops, G)
        # For now, sum motif id's in k-hop (min is 0 for no motifs)
        motif_in_khop = torch.sum(torch.unique([G.nodes[ni]['motif_id'] for ni in nodes_in_khop])).item()
        return torch.tensor(motif_in_khop, dtype=torch.long)

    return get_label

def binary_feature_label(G, method = 'median'):
    '''
    Labeling based solely on features, no edge information
        - Keywords can be given based on type of labeling split

    Args:
        G (nx.Graph): Graph on which the nodes are labeled on
        method (str): Method by which to split the features
    '''
    max_node = len(list(G.nodes))
    node_attr = nx.get_node_attributes(G, 'x')
    if method == 'median':
        x1 = [node_attr[i][1] for i in range(max_node)]
        split = torch.median(x1).item()
    def get_label(node_idx):
        return torch.tensor(int(x1[node_idx] > split), dtype=torch.long)

    return get_label

def number_motif_equal_label(G, num_hops, equal_number=1):
    def get_label(node_idx):
        nodes_in_khop = khop_subgraph_nx(node_idx, num_hops, G)
        num_unique_houses = torch.unique([G.nodes[ni]['shape'] \
            for ni in nodes_in_khop if G.nodes[ni]['shape'] > 0 ]).shape[0]
        return torch.tensor(int(num_unique_houses == equal_number), dtype=torch.long)

    return get_label

def bound_graph_label(G: nx.Graph):
    '''
    Args:
        G (nx.Graph): Graph on which the labels are based on
    '''
    sh = nx.get_node_attributes(G, 'shapes_in_khop')
    def get_label(node_idx):
        return torch.tensor(sh[node_idx] - 1, dtype=torch.long)

    return get_label

def logical_edge_feature_label(G, num_hops = None, feature_method = 'median'):

    if feature_method == 'median':
        # Calculate median (as for feature):
        node_attr = nx.get_node_attributes(G, 'x')
        x1 = [node_attr[i][1] for i in range(G.number_of_nodes())]
        split = torch.median(x1).item()

    def get_label(node_idx):
        nodes_in_khop = khop_subgraph_nx(node_idx, num_hops, G)
        num_unique_houses = torch.unique([G.nodes[ni]['shape'] \
            for ni in nodes_in_khop if G.nodes[ni]['shape'] > 0 ]).shape[0]
        return torch.tensor(int(num_unique_houses == 1 and x1[node_idx] > split), dtype=torch.long)

    return get_label
    
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_predict

class ShapeGGen(NodeDataset):
    '''
    Full ShapeGGen dataset implementation

    ..note:: Flag and circle shapes not yet implemented
    
    Args:
        model_layers (int, optional): Number of layers within the GNN that will
            be explained. This defines the extent of the ground-truth explanations
            that are created by the method. (:default: :obj:`3`)
        shape (str, optional): Type of shape to be inserted into graph.
            Options are `'house'`, `'flag'`, `'circle'`, and `'multiple'`. 
            If `'multiple'`, random shapes are generated to insert into 
            the graph.
            (:default: :obj:`'house'`)
        seed (int, optional): Seed for graph generation. (:default: `None`)

        TODO: Ensure seed keeps graph generation constant.

        kwargs: Additional arguments

            Graph Construction:
                variant (int): 0 indicates using the old ShapeGGen method, and 1 indicates
                    using the new ShapeGGen method (i.e. one with pref. attachment).   
                num_subgraphs (int): Number of individual subgraphs to use in order to build
                    the graph. Doesn't guarantee size of graph. (:default: :obj:`10`)
                prob_connection (float): Probability of making a connection between any two
                    of the original subgraphs. Roughly controls sparsity and number of 
                    class 0 vs. class 1 nodes. (:default: :obj:`1`)
                subgraph_size (int): Expected size of each individual subgraph.
                base_graph (str): Base graph structure to use for generating subgraphs.
                    Only in effect for variant 0. (:default: :obj:`'ba'`)
                verify (bool): Verifies that graph does not have any "bad" motifs in it.
                    (:default: :obj:`True`)
                max_tries_verification (int): Maximum number of tries to re-generate a 
                    graph that contains bad motifs. (:default: :obj:`5`)

            Feature attribution:
                n_informative (int): Number of informative features, i.e. those that
                    are correlated with label. (:default: :obj:`4`)
                class_sep (float):
                n_features (int):
                n_clusters_per_class (int):
                homophily_coef (float):

            Sensitive feature:
                add_sensitive_feature (bool):  Whether to include a sensitive, discrete 
                    attribute in the node features. If this is true, the total number of
                    features will be `n_features + 1`. (:default: :obj:`True`) 
                attribute_sensitive_feature (bool): Whether to attribute the sensitive
                    feature to the label of the dataset. `False` means to generate
                    sensitive features randomly (i.e. uncorrelated). 
                    (:default: :obj:`False`)
                sens_attribution_noise (float):
            
    Members:
        G (nx.Graph): Networkx version of the graph for the dataset
            - Contains many values per-node: 
                1. 'shape': which motif a given node is within
                2. 'shapes_in_khop': number of motifs within a (num_hops)-
                    hop neighborhood of the given node.
                    - Note: only for bound graph
    '''

    def __init__(self,  
        model_layers: int = 3,
        shape: Union[str, nx.Graph] = 'house',
        seed: Optional[int] = None,
        make_explanations: Optional[bool] = True,
        **kwargs): # TODO: turn the last three arguments into kwargs

        super().__init__(name = 'ShapeGGen', num_hops = model_layers)

        self.in_shape = []
        self.graph = None
        self.model_layers = model_layers
        self.make_explanations = make_explanations

        # Parse kwargs:
        self.variant = 1 if 'variant' not in kwargs else kwargs['variant']
            # 0 is old, 1 is preferential attachment one
        self.num_subgraphs = 10 if 'num_subgraphs' not in kwargs else kwargs['num_subgraphs']
        self.prob_connection = 1 if 'prob_connection' not in kwargs else kwargs['prob_connection']
        self.subgraph_size = 13 if 'subgraph_size' not in kwargs else kwargs['subgraph_size']
        self.base_graph = 'ba' if 'base_graph' not in kwargs else kwargs['base_graph']
        self.verify = True if 'verify' not in kwargs else kwargs['verify']
        self.max_tries_verification = 5 if 'max_tries_verification' not in kwargs else kwargs['max_tries_verification']

        # Feature args:
        self.n_informative = 4 if 'n_informative' not in kwargs else kwargs['n_informative']
        self.class_sep = 1.0 if 'class_sep' not in kwargs else kwargs['class_sep']
        self.n_features = 10 if 'n_features' not in kwargs else kwargs['n_features']
        # Note: n_clusters_per_class assumed to be 2 for the publication
        self.n_clusters_per_class = 2 if 'n_clusters_per_class' not in kwargs else kwargs['n_clusters_per_class']
        self.homophily_coef = None if 'homophily_coef' not in kwargs else kwargs['homophily_coef']

        # Sensitive feature:
        self.add_sensitive_feature = True if 'add_sensitive_feature' not in kwargs else kwargs['add_sensitive_feature']
        self.attribute_sensitive_feature = False if 'attribute_sensitive_feature' not in kwargs else kwargs['attribute_sensitive_feature']
        self.sens_attribution_noise = 0.25 if 'sens_attribution_noise' not in kwargs else kwargs['sens_attribution_noise']

        self.seed = seed

        # Get shape:
        self.shape_method = ''
        if isinstance(shape, nx.Graph):
            self.insert_shape = shape
        else:
            self.insert_shape = None
            shape = shape.lower()
            self.shape_method = shape
            if shape == 'house':
                self.insert_shape = house
            elif shape == 'flag':
                pass
            elif shape == 'circle':
                self.insert_shape = pentagon # 5-member ring
            assert shape != 'random', 'Multiple shapes not yet supported for bounded graph'

        # Build graph:

        if self.verify and shape != 'random':
            for i in range(self.max_tries_verification):
                if self.variant == 0:
                    self.G = BBG_old(
                        shape = self.insert_shape, 
                        num_subgraphs = self.num_subgraphs, 
                        inter_sg_connections = 1,
                        prob_connection = self.prob_connection,
                        subgraph_size = self.subgraph_size,
                        num_hops = 1,
                        base_graph = self.base_graph,
                        seed = self.seed,
                        )

                elif self.variant == 1:
                    self.G = BBG_PA(
                        shape = self.insert_shape, 
                        num_subgraphs = self.num_subgraphs, 
                        inter_sg_connections = 1,
                        prob_connection = self.prob_connection,
                        subgraph_size = self.subgraph_size,
                        num_hops = 1,
                        seed = self.seed
                        )

                if verify_motifs(self.G, self.insert_shape):
                    # If the motif verification passes
                    break
            else:
                # Raise error if we couldn't generate a valid graph
                raise RuntimeError(f'Could not build a valid graph in {self.max_tries_verification} attempts. \
                    \n Try using different parameters for graph generation or increasing max_tries_verification argument value.')
            
        else:
            if self.variant == 0:
                self.G = BBG_old(
                        shape = self.insert_shape, 
                        num_subgraphs = self.num_subgraphs, 
                        inter_sg_connections = 1,
                        prob_connection = self.prob_connection,
                        subgraph_size = self.subgraph_size,
                        num_hops = 1,
                        base_graph = self.base_graph,
                        seed = self.seed,
                        )
            elif self.variant == 1:
                self.G = BBG_PA(
                    shape = self.insert_shape, 
                    num_subgraphs = self.num_subgraphs, 
                    inter_sg_connections = 1,
                    prob_connection = self.prob_connection,
                    subgraph_size = self.subgraph_size,
                    num_hops = 1,
                    seed = self.seed
                    )

        self.num_nodes = self.G.number_of_nodes() # Number of nodes in graph
        self.generate_shape_graph() # Performs planting, augmenting, etc.

        # Set random splits for size n graph:
        range_set = list(range(self.num_nodes))
        random.seed(1234) # Seed random before making splits
        train_nodes = random.sample(range_set, int(self.num_nodes * 0.7))
        test_nodes  = random.sample(range_set, int(self.num_nodes * 0.25))
        valid_nodes = random.sample(range_set, int(self.num_nodes * 0.05))

        self.fixed_train_mask = torch.tensor([s in train_nodes for s in range_set], dtype=torch.bool)
        self.fixed_test_mask = torch.tensor([s in test_nodes for s in range_set], dtype=torch.bool)
        self.fixed_valid_mask = torch.tensor([s in valid_nodes for s in range_set], dtype=torch.bool)

    def generate_shape_graph(self):
        '''
        Generates the full graph with the given insertion and planting policies.

        :rtype: :obj:`torch_geometric.Data`
        Returns:
            data (torch_geometric.Data): Entire generated graph.
        '''

        gen_labels = bound_graph_label(self.G)
        y = torch.tensor([gen_labels(i) for i in self.G.nodes], dtype=torch.long)
        self.yvals = y.detach().clone() # MUST COPY TO AVOID MAJOR BUGS

        gen_features, self.feature_imp_true = gaussian_lv_generator(
            self.G, self.yvals, seed = self.seed,
            n_features = self.n_features,
            class_sep = self.class_sep,
            n_informative = self.n_informative,
            n_clusters_per_class=self.n_clusters_per_class,
        )
        x = torch.stack([gen_features(i) for i in self.G.nodes]).float()

        if self.add_sensitive_feature:

            # Choose sensitive feature randomly
            if self.seed is not None:
                torch.manual_seed(self.seed)

            if self.attribute_sensitive_feature:
                print('Adding sensitive attr')
                prob_change = (torch.rand((y.shape[0],)) < self.sens_attribution_noise)
                sensitive = torch.where(prob_change, torch.logical_not(y.bool()).long(), y).float()
            else:
                sensitive = torch.randint(low=0, high=2, size = (x.shape[0],)).float()

            # Add sensitive attribute to last dimension on x
            x = torch.cat([x, sensitive.unsqueeze(1)], dim = 1)
            # Expand feature importance and mark last dimension as negative
            self.feature_imp_true = torch.cat([self.feature_imp_true, torch.zeros((1,))])

            # Shuffle to mix in x:
            shuffle_ind = torch.randperm(x.shape[1])
            x[:,shuffle_ind] = x.clone()
            self.feature_imp_true[shuffle_ind] = self.feature_imp_true.clone()

            # Sensitive feature is in the location where the last index was:
            self.sensitive_feature = shuffle_ind[-1].item()

        else:
            self.sensitive_feature = None

        edge_index = to_undirected(torch.tensor(list(self.G.edges), dtype=torch.long).t().contiguous())

        if self.homophily_coef is not None:
            feat_mask = torch.logical_not(self.feature_imp_true)
            if self.sensitive_feature is not None:
                feat_mask[self.sensitive_feature] = False

            x = optimize_homophily(
                x = x,
                edge_index = edge_index,
                label = y,
                feature_mask = feat_mask,
                homophily_coef = self.homophily_coef,
                epochs = 1000,
                connected_batch_size = (edge_index.shape[1] // 2),
                disconnected_batch_size = math.comb(self.num_nodes, 2) // self.num_nodes
            )

        for i in sorted(self.G.nodes):
            self.G.nodes[i]['x'] = x[i,:].detach().clone() #gen_features(i)

        self.graph = Data(
            x=x, 
            y=y,
            edge_index = edge_index, 
            shape = torch.tensor(list(nx.get_node_attributes(self.G, 'shape').values()))
        )

        # Generate explanations:
        if self.make_explanations:
            self.explanations = [self.explanation_generator(n) for n in sorted(self.G.nodes)]
        else:
            self.explanations = None

    def explanation_generator(self, node_idx):

        # Label node and edge imp based off of each node's proximity to a house

        # Find nodes in num_hops
        original_in_num_hop = set([self.G.nodes[n]['shape'] for n in khop_subgraph_nx(node_idx, 1, self.G) if self.G.nodes[n]['shape'] != 0])

        # Tag all nodes in houses in the neighborhood:
        khop_nodes = khop_subgraph_nx(node_idx, self.model_layers, self.G)
        node_imp_map = {i:(self.G.nodes[i]['shape'] in original_in_num_hop) for i in khop_nodes}
            # Make map between node importance in networkx and in pytorch data

        khop_info = k_hop_subgraph(
            node_idx,
            num_hops = self.model_layers,
            edge_index = to_undirected(self.graph.edge_index)
        )

        node_imp = torch.tensor([node_imp_map[i.item()] for i in khop_info[0]], dtype=torch.double)

        # Get edge importance based on edges between any two nodes in motif
        in_motif = khop_info[0][node_imp.bool()] # Get nodes in the motif
        edge_imp = torch.zeros(khop_info[1].shape[1], dtype=torch.double)
        for i in range(khop_info[1].shape[1]):
            # Highlight edge connecting two nodes in a motif
            if (khop_info[1][0,i] in in_motif) and (khop_info[1][1,i] in in_motif):
                edge_imp[i] = 1
                continue
            
            # Make sure that we highlight edges connecting to the source node if that
            #   node is not in a motif:
            one_edge_in_motif = ((khop_info[1][0,i] in in_motif) or (khop_info[1][1,i] in in_motif))
            node_idx_in_motif = (node_idx in in_motif)
            one_end_of_edge_is_nidx = ((khop_info[1][0,i] == node_idx) or (khop_info[1][1,i] == node_idx))

            if (one_edge_in_motif and one_end_of_edge_is_nidx) and (not node_idx_in_motif):
                edge_imp[i] = 1

        exp = Explanation(
            feature_imp=self.feature_imp_true,
            node_imp = node_imp,
            edge_imp = edge_imp,
            node_idx = node_idx
        )
        exp.set_enclosing_subgraph(khop_info)

        # Return list of single element since ShapeGGen produces unique explanations
        return exp


    def visualize(self, shape_label = False, ax = None, show = False):
        '''
        Args:
            shape_label (bool, optional): If `True`, labels each node according to whether
            it is a member of an inserted motif or not. If `False`, labels each node 
            according to its y-value. (:default: :obj:`True`)
        '''

        ax = ax if ax is not None else plt.gca()

        Gitems = list(self.G.nodes.items())
        node_map = {Gitems[i][0]:i for i in range(self.G.number_of_nodes())}

        if shape_label:
            y = [int(self.G.nodes[i]['shape'] > 0) for i in range(self.num_nodes)]
        else:
            ylist = self.graph.y.tolist()
            y = [ylist[node_map[i]] for i in self.G.nodes]

        node_weights = {i:node_map[i] for i in self.G.nodes}

        #pos = nx.kamada_kawai_layout(self.G)
        pos = nx.spring_layout(self.G, seed = 1234) # Seed to always be consistent in output
        #_, ax = plt.subplots()
        nx.draw(self.G, pos, node_color = y, labels = node_weights, ax=ax)
        #ax.set_title('ShapeGGen')
        #plt.tight_layout()

        if show:
            plt.show()

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

def get_data(num):
    '''This function allows us to load our data. We use the supergraph data - the original graph plus
        extra points associated with different points that are highly correlated to see if our explainers
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

sparsity = [sys.argv[1]]
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
    lst = []
    ru = 25
    ep = 50
    set_seed(0)
    batch_lst = []
    for i in range(0, ru):
        if sys.argv[2] == 'Beta':
            print('Beta!')
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
            lst.append([sparse, 'Beta', i, ep, runtime, avg_runtime])
        elif sys.argv[1] == 'SubgraphX':
            if int(sys.argv[1]) == 25:
                rollout = 20
                min_atoms = 1
                c_puct = 8.165129763712843
                expand_atoms = 5
                sample_num = 6
            else:
                rollout = 4
                min_atoms = 1
                c_puct = 2.0727783316948987
                expand_atoms = 7
                sample_num = 9
            start = time.time()
            expgnn = explainer.get_explanation_graph(data.x, data.edge_index, data.y) 
            prediction_mask = expgnn.edge_imp.numpy()
            end = time.time()
            runtime = end - start
            avg_runtime = runtime / ep
            print(f'GNN, Sparsity: {sparse}, Runtime: {runtime}, Average: {(end - start) / ep}')
            lst.append([sparse, 'SubgraphX', i, ep, runtime, avg_runtime])
        else:
            print('GNN!')
            ep = 25
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
            loader = graph_data
            bat = 0
            for batch in loader:
                if bat != 0:
                    break
                else:
                    x = torch.reshape(batch.x, (batch.x.shape[0], 1))
                    explanation = explainer(x, batch.edge_index, target=batch.y)
                    end = time.time()
                    bat += 1
            runtime = end - start
            avg_runtime = runtime / ep
            print(f'GNN, Sparsity: {sparse}, Runtime: {runtime}, Average: {(end - start) / ep}')
            lst.append([sparse, 'GNN', i, ep, runtime, avg_runtime])
    df = pd.DataFrame(lst, columns=['Data', 'Explainer', 'Experiment', '# Epochs', 'Full Runtime', 'Average Runtime'])
    df.to_csv(f'time/SERGIO{sparse}{sys.argv[2]}Runtimes.csv')