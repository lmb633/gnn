import os.path as osp
from torch_geometric.datasets import Planetoid
from torch_geometric.data import NeighborSampler
import torch_geometric.transforms as T
from gcn_test.models import GCNNet
import torch
from torch import nn
import numpy as np

embedding = 16
lr = 0.001
epoch = 300

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]
print(dataset.num_features)
print(dataset.num_classes)
print(data)
print(data.num_nodes)
print(data.edge_attr)


def validate(out, label):
    right = 0
    for pair in zip(out, label):
        idx = np.argmax(pair[0].detach().numpy())
        if idx == pair[1]:
            right += 1
    return right


# dataloader = NeighborSampler(data, size=[20, 20], num_hops=2, batch_size=2, shuffle=True, add_self_loops=True)

model = GCNNet(dataset.num_features, embedding, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.NLLLoss()
for i in range(epoch):
    optimizer.zero_grad()
    out = model(data)
    # print(out.shape)
    # print(data.y.shape)
    loss = criterion(out, data.y)
    print(loss)
    print(validate(out,data.y))
    loss.backward()
    optimizer.step()
