import torch
from torch_geometric.nn.conv import SAGEConv
import torch.nn.functional as F


class SAGENet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAGENet, self).__init__()
        self.conv1 = SAGEConv(in_channels, 16, normalize=False)
        self.conv2 = SAGEConv(16, out_channels, normalize=False)

    def forward(self, x, data_flow):
        # for i in range(len(data_flow)):
        data = data_flow[0]
        # print(data.n_id)
        x = x[data.n_id]
        x = F.relu(self.conv1((x, None), data.edge_index, size=data.size))
        x = F.dropout(x, p=0.5, training=self.training)

        data = data_flow[1]
        # print(data.n_id)
        x = self.conv2((x, None), data.edge_index, size=data.size)
        return F.log_softmax(x, dim=1)
