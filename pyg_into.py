# @Author  : Edlison
# @Date    : 3/2/23 21:42
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='data/cora', name='cora')


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, training=self.training)
        h = self.conv2(h, edge_index)

        return F.log_softmax(h, dim=1)


if __name__ == '__main__':
    # device = torch.device('cuda')
    # model = GCN().to(device)
    # data = dataset[0].to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    #
    # model.train()
    # for epoch in range(200):
    #     optimizer.zero_grad()
    #     out = model(data)
    #     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    #     loss.backward()
    #     optimizer.step()
    #
    # model.eval()
    # pred = model(data).argmax(dim=1)
    # cor = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    # acc = cor / data.test_mask.sum()
    # print('acc: ', acc)
    data = dataset[0]
    print(data.num_node_features)
    print(dataset.num_classes)
