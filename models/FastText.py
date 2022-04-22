# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastTextModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dropout, num_classes):
        super(FastTextModel, self).__init__()
        self.embeding = nn.Embedding(num_embeddings, embedding_dim)
        self.droupout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        out = self.embeding(x)

        # # SpatialDroupout
        # out = out.permute(0, 2, 1)  # convert to [batch, channels, time]
        # out = F.dropout2d(out, 0.2, training=True)
        # out = out.permute(0, 2, 1)  # back to [batch, time, channels]

        # out = out.mean(dim=1)
        out = out.max(dim=1).values
        out = self.fc1(out)
        out = F.relu(out)
        out = self.droupout(out)
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    net = FastTextModel(10000, 300, 0.2, 5)
    print(net)
    # from torchsummary import summary
    # summary(net, (20,))
    x = torch.randint(0, 10000, (1, 20))
    y = net(x)
    print(y)
