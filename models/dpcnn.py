# encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DPCNNModel(nn.Module):
    def __init__(self,
                 num_filters,
                 num_classes,
                 num_embeddings=None, embedding_dim=None,
                 embedding_matrix=None, freeze=False):
        super(DPCNNModel, self).__init__()
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix,
                                                          freeze=freeze,
                                                          padding_idx=embedding_matrix.shape[0] - 1)
        else:
            self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=num_embeddings - 1)
        self.embedding_dim =embedding_dim
        if embedding_matrix is not None:
            self.embedding_dim = embedding_matrix.shape[1]
        self.conv_region = nn.Conv2d(1, num_filters, (3, self.embedding_dim), stride=(1, 1))
        self.conv = nn.Conv2d(num_filters, num_filters, (3, 1), stride=(1, 1))
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        x, _ = x
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.conv_region(x)

        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)

        while x.size()[2] >= 2:
            x = self._block(x)
        x = x.squeeze()
        x = self.fc(x)
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        x = x + px
        return x


if __name__ == '__main__':
    net = DPCNNModel(256, 15, num_embeddings=10000, embedding_dim=300)

    # # need rm Embedding Layer
    # from torchsummary import summary
    # summary(net, (20, 300))
    # exit(0)

    # # need rm Embedding layer
    # from torchinfo import summary
    # summary(net, (32, 20, 300))
    # exit(0)

    print(net)
    x = torch.randint(0, 10000, (32, 20))
    y = net((x, None))
    print(y)