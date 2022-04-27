# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


'''Bag of Tricks for Efficient Text Classification'''


class FastTextModel(nn.Module):
    def __init__(self, dropout, num_classes,
                 embedding_matrix=None, freeze=False,
                 num_embeddings=None, embedding_dim=None):
        super(FastTextModel, self).__init__()
        if embedding_matrix:
            self.embeding = nn.Embedding.from_pretrained(embedding_matrix,
                                                         freeze=freeze,
                                                         padding_idx=embedding_matrix.shape[0]-1)
        else:
            self.embeding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=num_embeddings-1)
        self.droupout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x, seq_lens = x
        x = self.embeding(x)

        # # SpatialDroupout
        # out = out.permute(0, 2, 1)  # convert to [batch, channels, time]
        # out = F.dropout2d(out, 0.2, training=True)
        # out = out.permute(0, 2, 1)  # back to [batch, time, channels]

        out = x.mean(dim=1)
        # out = out.max(dim=1).values
        out = self.fc1(out)
        out = F.relu(out)
        out = self.droupout(out)
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    net = FastTextModel(0.2, 5, num_embeddings=10000, embedding_dim=300)
    print(net)

    # # need rm Embedding layer
    # from torchsummary import summary
    # summary(net, (20, 300))
    # exit(0)

    # # need rm Embedding layer
    # from torchinfo import summary
    # summary(net, (32, 20, 300))
    # exit(0)

    x = torch.randint(0, 10000, (32, 20))
    y = net((x, None))
    print(y)
