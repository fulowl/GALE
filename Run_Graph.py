import torch
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from eval.eval import get_split, SVMEvaluator
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.loader import DataLoader

from utils.tu_data import TUDataset
from utils.loss import batch_loss


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv in self.layers:
            z = conv(z, edge_index)
            z = self.activation(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        g = torch.cat(gs, dim=1)
        return z, g


class FC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x) + self.linear(x)


class Encoder(torch.nn.Module):
    def __init__(self, gcn, mlp1, mlp2):
        super(Encoder, self).__init__()
        self.gcn = gcn
        self.mlp1 = mlp1
        self.mlp2 = mlp2


    def forward(self, x, edge_index, batch):
        z, g = self.gcn(x, edge_index, batch)
        h = self.mlp1(z)
        g = self.mlp2(g)
        return h, g


def train(encoder_model, dataloader, optimizer):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to('cuda')
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        h, g = encoder_model(data.x, data.edge_index, data.batch)
        
        if True:
            # Use PageRank
            loss = batch_loss(h, data.pr, feat=data.x, batch=data.batch)
        else:
            # Use AE
            loss = batch_loss(h, data.orbit_label, feat=data.x, batch=data.batch)
        
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss


def test(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        _, g = encoder_model(data.x, data.edge_index, data.batch)
        x.append(g)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    result = SVMEvaluator(linear=True)(x, y, split)
    return result

def main():
    device = torch.device('cuda')
    path = 'datasets'
    dataset = TUDataset(path, name='MUTAG')

    dataloader = DataLoader(dataset, batch_size=16)
    for i, batch in enumerate(dataloader):
        assert batch.pr.shape[0] == batch.batch.shape[0]
    input_dim = max(dataset.num_features, 1)

    gcn = GConv(input_dim=input_dim, hidden_dim=512, num_layers=2).to(device)
    mlp1 = FC(input_dim=512, output_dim=512)
    mlp2 = FC(input_dim=512 * 2, output_dim=512)
    encoder_model = Encoder(gcn=gcn, mlp1=mlp1, mlp2=mlp2).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.01)

    with tqdm(total=100, desc='(T)') as pbar:
        for epoch in range(1, 101):
            loss = train(encoder_model, dataloader, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

            if epoch % 5 == 0:
                test_result = test(encoder_model, dataloader)
                print(f'(E): Best test ACC={test_result["acc"]:.4f}, F1Mi={test_result["micro_f1"]:.4f}')
                


if __name__ == '__main__':
    main()
