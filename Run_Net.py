import torch
import torch_geometric.transforms as T
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from eval.eval import get_split, LREvaluator
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import uniform
from torch_geometric.datasets import Planetoid


from utils.loss import loss_net
from torch_geometric.utils import to_networkx
import networkx as nx
import pynauty as nauty



class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index, edge_weight):
        return x[torch.randperm(x.size(0))], edge_index, edge_weight

    def forward(self, x, edge_index, edge_weight=None):
        z = self.encoder(x, edge_index, edge_weight)

        return z


def train(encoder_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z = encoder_model(data.x, data.edge_index)
    
    loss = loss_net(z, data.pr, data.x, data.orbits, use_pr=False)

    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    z = encoder_model(data.x, data.edge_index)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, data.y, split)
    return result


def pyg_to_pynauty(data):
    n = data.num_nodes
    edge_index = data.edge_index.numpy()
    edges = list(zip(edge_index[0], edge_index[1]))

    G = nauty.Graph(number_of_vertices=n, directed=False)
    for u, v in edges:
        G.connect_vertex(u, v)
    return G


def add_pr_orbit(data):
    # Compute PageRank
    G = to_networkx(data, to_undirected=True)
    pagerank = nx.pagerank(G, alpha=0.85)
    data.pr = torch.tensor([v for _, v in sorted(pagerank.items())], dtype=torch.float)

    # Compute AE
    G_ = pyg_to_pynauty(data.cpu().detach())
    _, _, _, orbits, _ = nauty.autgrp(G_)
    data.orbits = torch.tensor(orbits)

    for (node, value), tensor_value in zip(sorted(pagerank.items()), data.pr):
        assert abs(value - tensor_value.item()) < 1e-6, "Error."
    
    return data.to('cuda')

def main():
    device = torch.device('cuda')
    path = 'datasets'

    dataset = Planetoid(path, name='Cora', transform=T.NormalizeFeatures())

    data = dataset[0].to(device)
    
    data = add_pr_orbit(data)
    
    gconv = GConv(input_dim=dataset.num_features, hidden_dim=256, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, hidden_dim=256).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.001)

    with tqdm(total=5000, desc='(T)') as pbar:
        for epoch in range(1, 5001):
            loss = train(encoder_model, data, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()
            
            if epoch % 5 == 0:
                test_result = test(encoder_model, data)
                print(f'(E): Best test ACC={test_result["acc"]:.4f}, F1Mi={test_result["micro_f1"]:.4f}')


if __name__ == '__main__':
    main()
