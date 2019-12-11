from torch_geometric.nn.conv import GCNConv
import torch
import torch.nn.functional as F


class GCNNet(torch.nn.Module):
    def __init__(self, inchannel, embedding, outchannel):
        super(GCNNet, self).__init__()
        self.gcn1 = GCNConv(inchannel, embedding, cached=True)
        self.gcn2 = GCNConv(embedding, outchannel, cached=True)

    def forward(self, x):
        x, edge_index, edge_weight = x.x, x.edge_index, x.edge_attr
        x = F.relu(self.gcn1(x, edge_index, edge_weight))
        x = F.dropout(x)
        x = self.gcn2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)
