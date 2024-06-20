# This file generates an edge graph based on SERGIO features that are highly correlated.
# It takes in two inputs: the level of sparsity of the SERGIO features and correlation level (elements must be at least as correlated as the value).
import pandas as pd
import numpy as np
from scipy import stats
import torch
import sys

def get_data(val):
    '''
        This function returns the ground truth matrix for the SERGIO data as well as the feature and label matrices. It will return the matrices as well as the
        number of features and classes.
    '''
    features = np.load(f'sergio data/SERGIOsimu_{val}Sparse_noLibEff_concatShuffled.npy')
    num_features = features.shape[1]
    labels = np.load(f'sergio data/SERGIOsimu_{val}Sparse_noLibEff_cTypes.npy')
    num_classes = len(np.unique(labels))
    adj = np.load('sergio data/gt_adj.npy')
    return adj, features, labels, num_features, num_classes

# Return the adjacency and feature matrix with all important data; ensure that the matrices are in numpy array format.
adj, features, labels, num_features, num_classes = get_data(sys.argv[1])
edge_index = np.array(adj)
features = np.array(features)

def create_comp_graph(feat, edges, corr):
    '''
        Based on a given correlation (corr) and feature matrix (feat), this function returns an edge matrix in the form of a 2 x number of edges array of the
        edges generated based on correlated features. Self-correlated edges are not included.
    '''
    X = feat.T # Transform the feature matrix to be gene-by-cell as we want correlated genes for edges, not correlated cells.
    edge_index = edges # Copy ground truth over.
    interactions = {} # Initialize record of correlation for each gene-gene combination.
    
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            interactions[(i, j)] = abs(stats.pearsonr(X[i, :], X[j, :])[0]) # Get correlation for each combination of genes.
            
    gt_edges = set() # Initialize set of edges in the ground-truth.
    for i in range(edges.shape[1]):
        e = (edge_index[0][i].item(), edge_index[1][i].item()) # Get each node for an edge in the groundtruth.
        gt_edges.add(e) # Add the edge to the groundtruth set.
        
    # The following two lists will form the basis for the new 2xedge matrix for the correlations-based edge index.
    s_l = [] # Initialize first node list
    d_l  = [] # Initialize second node list
    overlap = 0 # Count the number of edges in both the new edge set and groundtruth edge set to ensure there's some overlap.
    for (s, d), i in interactions.items(): # Iterate over all possible edges and associated correlations.
        if i >= corr and s != d: # Ensure the edge is at least as correlated as our baseline correlation and the correlation isn't self-correlation to add to record.
            s_l.append(s)
            d_l.append(d)
        else: # Otherwise, ignore the edge.
            continue
            
    print(f"Ne: {len(s_l)} | Ne_GT: {ne} | Overlap: {overlap}") # Return the number of edges in the new edge index, in the groundtruth, and in both to check results. 
    edge_index = np.array([s_l, d_l]) # Convert two lists into edge array to be returned
    return edge_index

res = create_comp_graph(features, edge_index, float(sys.argv[2])) # Create edge index based on data and given correlation.
np.save(f'ExtraPointsSergio{sys.argv[1]}.npy', res) # Save to be used later for analysis.