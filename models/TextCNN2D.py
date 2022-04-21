#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN2DModel(nn.Module):
    def __init__(self,
                 n_vocab, embedding_dim,
                 num_filters,
                 filter_sizes,
                 dropout,
                 num_classes,
                 embedding_pretrained=None):
        super(TextCNN2DModel, self).__init__()
        if embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(n_vocab, embedding_dim, padding_idx=n_vocab-1)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(1, num_filters, (k, embedding_dim)) for k in filter_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    net = TextCNN2DModel(10000, 300, 256, (2, 3, 4), 0.2, 15)

    # # need rm Embedding layer
    # from torchsummary import summary
    # summary(net, (20, 300))
    # exit(0)

    print(net)
    x = torch.randint(0, 10000, (1, 20))
    y = net(x)
    print(y)