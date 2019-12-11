from torch_geometric.datasets import KarateClub
from torch_geometric.data import DataLoader
from sageconv_test.models import SAGENet
from torch import nn
import torch
from utils import clip_gradient, save_checkpoint, AverageMeter
from torch_geometric.data import Data
from torch_geometric.data import NeighborSampler

epoch = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.001
grad_clip = 5.0
print_freq = 1


def validation(result):
    right = 0
    for pairs in result:
        for label, out in zip(pairs[0], pairs[1]):
            if label == 0 and out[0] > out[1]:
                right += 1
            if label == 1 and out[0] < out[1]:
                right += 1
    return right


def train_net():
    club = KarateClub()
    data = club.data
    data.num_nodes = data.num_nodes[0]
    print(data)
    data_loader = NeighborSampler(data, size=[20, 10], num_hops=2, batch_size=8, shuffle=True, add_self_loops=True)
    net = SAGENet(34, 2)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    losses = AverageMeter()
    for i in range(epoch):
        result = []
        for data_flow in data_loader():
            label = data.y[data_flow.n_id]
            # print(label)
            optimizer.zero_grad()
            out = net(data.x, data_flow.to(device))

            # print(out)
            # print(label)
            result.append((label, out))
            # Calculate loss
            loss = criterion(out, label)

            # Back prop.
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            clip_gradient(optimizer, grad_clip)

            # Update weights
            optimizer.step()
            losses.update(loss.item())

            # Print status
        print(validation(result))
        if i % print_freq == 0:
            print('Epoch: {0} Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(i, loss=losses))

    return losses.avg


if __name__ == '__main__':
    train_net()
