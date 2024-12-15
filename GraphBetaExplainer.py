import torch
import torch.nn.functional as F
import torch.distributions.constraints as constraints
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO

class BetaExplainer:
    '''This is a post-hoc explainer based on the Beta Distribution.'''
    def __init__(self, model: torch.nn.Module, data: torch.Tensor, G: torch.Tensor, device: torch.device, sz, a, b):
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
        