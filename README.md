# BetaExplainerDemo
## Description
We present BetaExplainer, a new GNN explainer method. This formulation takes advantage of probabalisitic learning to convey uncertainty in edge importance and provide prior information on edge importance. It outperforms GNNExplainer and SubgraphX on the unfaithfulness metric for challenging datasets. Our model is demonstrated below:

![image](https://github.com/wsloneker/BetaExplainerDemo/blob/main/BetaExplainerDemo.png)

## Requirements
Python: 3.12.7

PyTorch Version: 2.7.0

PytorchGeometric: 2.6

Pyro-PPL: 1.9.1

To install requirements:

pip install -r requirements.txt

## Evaluation
We use unfaithfulness to measure explainer performance across datasets, as this metric can analyze explainer performance on datasets that may not neccessarily have a notion of groundtruth. Accuracy, precision, recall, and F1 Scores are calculated for those datsets that do have the notion of groundtruth. To determine results, run file RunBetaExplainer.py. 

To test the SERGIO datasets, use the following functions

SERGIO 25% Sparsity

python RunBetaExplainer.py 25 graph (layer) True (FinalLinearLayer)

SERGIO 50% Sparsity

python RunBetaExplainer.py 50 graph (layer) True (FinalLinearLayer)

These functions indicate the dataset, that graph classification is included, and that there's a notion of groundtruth. Replace the layer term with GCN, SAGE, GAT, or GATv2 to denote the respective convolutional layer of the model used. The script will determine ideal parameters for this model, as well as the BetaExplainer using Optuna. The (FinalLinearLayer) term denotes whether there will be a final linear term in the model or not using either True or False as a term.

For the ShapeGGen datasets, a similar method is used, with layers updatable as desired by the user, dataset modified to accurately select the dataset, and the classification problem type updated to node.

SG-BASE

python RunBetaExplainer.py base node (layer) True (FinalLinearLayer)

SG-UNFAIR

python RunBetaExplainer.py unfair node (layer) True (FinalLinearLayer)

SG-HETEROPHILIC

python RunBetaExplainer.py hetero node (layer) True (FinalLinearLayer)

SG-LESSINFORM

python RunBetaExplainer.py lessinform node (layer) True (FinalLinearLayer)

SG-MOREINFORM

python RunBetaExplainer.py moreinform node (layer) True (FinalLinearLayer)

Finally, we discuss a new dataset chosen by the user. This does not require a predefined model; just a predefinition of layer type. The python function is as follows:

python RunBetaExplainer.py (filename) (classification_problem) (layer) (notion_of_groundtruth) (FinalLinearLayer)

The notion_of_groundtruth input will take values True or False, based on whether the the "true" explanation is known or not. (layer) will indicate the layer architecture type taking values including GAT, GATv2, SAGE, or GCN. This indicates GATConv, GATv2Conv, SAGEConv, and GCNConv respectively. (classification_problem) takes either node or graph as its value, indicating whether node or graph classification is used. The (filename) component will suggest how to access the dataset.

It is assumed that filenames are denoted as follows: 'EdgeIndex{file_name}' (i. e. 'EdgeIndexOurs.npy' where file_name = 'Ours.npy'), 'Features{file_name}', and 'Labels{file_name}' for the edge index, features, and labels. It is assumed that the feature file will be node or graph by features. It is assumed that the labels will be accessed in column "Labels" and the edges will be denoted using two columns "P1" and "P2" denoting both endpoints in an edge if the files are denoted as a dataframe. Similarily, if the dataset has a known groundtruth adjacency matrix, this file will be accessed through file 'GTEdgeIndex{file_name}'. It is assumed the files will be of consistent format, one of a .npy, .csv, .tsv, or .xlsx file.

To replicate the baseline experiments, use ReplicateBetaExplainer.py. The commands will be python ReplicateBetaEXplainer.py (dataset). Replace (dataset) with 25, 50, Texas, base, unfair, hetero, lessinform, or moreinform to indicate the SERGIO 25% sparsity dataset, SERGIO 50% sparsity dataset, the Texas dataset, SG-BASE, SG-UNFAIR, SG-HETERO, SG-LESSINFORM, and SG-MOREINFORM respectively.