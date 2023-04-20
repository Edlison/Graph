# @Author  : Edlison
# @Date    : 3/5/23 11:02
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from gcn_pyg.models import Net


cuda = True if torch.cuda.is_available() else False


if __name__ == '__main__':
    device = torch.device('cuda')
    dataset = Planetoid(root='../data/cora', name='cora')
    data = dataset[0]
    model = Net(dataset.num_node_features, dataset.num_classes)
    if cuda:
        device = 1
        data.cuda(device)
        model.cuda(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for epoch in range(1):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        print('epoch: {}, loss: {:.4f}'.format(epoch, loss.item()))
        loss.backward()
        optimizer.step()

    model.eval()
    pred = model(data).argmax(dim=1)
    cor = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = cor / data.test_mask.sum()
    print('acc: ', acc)
