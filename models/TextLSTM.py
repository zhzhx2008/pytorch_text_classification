#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

class TextLSTMModel(nn.Module):
    def __init__(self,
                 n_vocab, embedding_dim,
                 hidden_size,
                 num_layers,
                 dropout,
                 num_classes,
                 embedding_pretrained=None):
        super(TextLSTMModel, self).__init__()
        if embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(n_vocab, embedding_dim, padding_idx=n_vocab-1)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        return out

if __name__ == '__main__':
    net = TextLSTMModel(10000, 300, 256, 2, 0.2, 15)

    # # need rm Embedding layer
    # from torchinfo import summary
    # summary(net, (5000, 20, 300))
    # exit(0)

    print(net)
    x = torch.randint(0, 10000, (1, 20))
    y = net(x)
    print(y)