# @Author  : Edlison
# @Date    : 3/5/23 10:43
import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import add_self_loops, degree


class GCN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # print('x shape: ', x.shape)
        # print('edge_index shape: ', edge_index.shape)

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # print('edge_index add self loops: ', edge_index.shape)

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)
        # print('x after lin: ', x.shape)

        # Step 3: Compute normalization.
        row, col = edge_index
        # print('row shape: ', row.shape)
        # print('col shape: ', col.shape)

        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)
        # print('hidden size: ', out.shape)

        # Step 6: Apply a final bias vector.
        out += self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        # print('x_j size: ', x_j.shape)
        # print('norm size: ', norm.shape)
        # print('norm.view(-1, 1): ', norm.view(-1, 1).shape)

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

    def aggregate(self, messages):
        return messages

    def update(self, aggr_out):
        return aggr_out


class Net(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.conv1 = GCN(num_node_features, 32)
        self.conv2 = GCN(32, 16)
        self.conv3 = GCN(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = self.conv1(x, edge_index)
        h = self.conv2(h, edge_index)
        h = self.conv3(h, edge_index)
        return F.log_softmax(h, dim=1)

