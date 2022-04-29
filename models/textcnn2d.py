#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


'''Convolutional Neural Networks for Sentence Classification'''


class TextCNN2DModel(nn.Module):
    def __init__(self,
                 num_filters,
                 filter_sizes,
                 dropout,
                 num_classes,
                 num_embeddings=None, embedding_dim=None, padding_idx=False,
                 embedding_matrix=None, freeze=False):
        super(TextCNN2DModel, self).__init__()
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix,
                                                          freeze=freeze,
                                                          padding_idx=padding_idx)
        else:
            self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.embedding_dim = embedding_matrix.shape[1] if embedding_matrix is not None else embedding_dim
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(1, num_filters, (k, self.embedding_dim)) for k in filter_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x, seq_lens = x
        x = self.embedding(x)
        out = x.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    net = TextCNN2DModel(256, (2, 3, 4), 0.2, 15, num_embeddings=10000, embedding_dim=300)

    # # need rm Embedding layer
    # from torchsummary import summary
    # summary(net, (20, 300))
    # exit(0)

    print(net)
    x = torch.randint(0, 10000, (32, 20))
    y = net((x, None))
    print(y)