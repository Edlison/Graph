# @Author  : Edlison
# @Date    : 3/5/23 23:03
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from gat_pyg.models import Net, NetConv


cuda = True if torch.cuda.is_available() else False


if __name__ == '__main__':
    dataset = Planetoid(root='../data/cora', name='cora')
    data = dataset[0]
    model = NetConv(dataset.num_node_features, dataset.num_classes)
    if cuda:
        device = 1
        data.cuda(device)
        model.cuda(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    iterations = 100

    model.train()
    for epoch in range(iterations):
        optimizer.zero_grad()
        out = model(data)

        # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

        print('epoch: {}, loss: {:.4f}'.format(epoch, loss.item()))
        loss.backward()
        optimizer.step()

    model.eval()
    pred = model(data).argmax(dim=1)
    cor = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = cor / data.test_mask.sum()
    print('acc: ', acc)  # 10epoch: 0.743; 20epoch: 0.744
