import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops


class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv, self).__init__(aggr='max')  # "Max" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.act = torch.nn.ReLU()
        self.update_lin = torch.nn.Linear(in_channels + out_channels, in_channels, bias=False)
        self.update_act = torch.nn.ReLU()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # print('forward=========')
        edge_index, _ = remove_self_loops(edge_index)
        # print('remove_self_loops', edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # print('add_self_loops', edge_index)
        # print('self.propagate1', edge_index, x.size(0), x.shape)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i,x_j):
        # x_j has shape [E, in_channels]
        # print('message============')
        # print('x_i', x_i.shape)
        # print('x_j', x_j.shape)
        x_j = self.lin(x_j)
        x_j = self.act(x_j)
        # print('x_j2', x_j.shape)
        return x_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]
        # print('update==============')
        # print('update', aggr_out.shape, x.shape)
        new_embedding = torch.cat([aggr_out, x], dim=1)

        new_embedding = self.update_lin(new_embedding)
        new_embedding = self.update_act(new_embedding)
        # print('new_embedding',new_embedding.shape)
        return new_embedding


if __name__ == '__main__':
    # from torch_geometric.utils import scatter_
    #
    # a = torch.FloatTensor([[8, 3, 6, 5, 1], [1, 3, 4, 5, 6], [1, 3, 8, 5, 8],[6, 3, 4, 5, 1],[1, 2, 3, 1, 6]])
    # print(scatter_('max', a, torch.tensor([0, 1, 0, 1, 2]), 0, 3))
    from torch_cluster import knn_graph

    x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    batch = torch.tensor([0, 0, 0, 0])
    edge_index = knn_graph(x, k=2, batch=batch, loop=False)
    print(edge_index)
