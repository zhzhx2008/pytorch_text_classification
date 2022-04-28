#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


'''Recurrent Convolutional Neural Networks for Text Classification'''


class TextRCNNModel(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_layers,
                 dropout,
                 num_classes,
                 seq_len,
                 num_embeddings=None, embedding_dim=None,
                 embedding_matrix=None, freeze=False):
        super(TextRCNNModel, self).__init__()
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix,
                                                          freeze=freeze,
                                                          padding_idx=embedding_matrix.shape[0]-1)
        else:
            self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=num_embeddings-1)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.maxpool = nn.MaxPool1d(seq_len)
        self.embed = embedding_matrix.shape[1] if embedding_matrix else embedding_dim
        self.fc = nn.Linear(hidden_size * 2 + self.embed, num_classes)

    def forward(self, x):
        x, _ = x
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = torch.cat((x, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out

if __name__ == '__main__':
    net = TextRCNNModel(256, 2, 0.2, 15, 20, num_embeddings=10000, embedding_dim=300)

    # # need rm Embedding layer
    # from torchinfo import summary
    # summary(net, (32, 20, 300))
    # exit(0)

    print(net)
    x = torch.randint(0, 10000, (32, 20))
    y = net((x, None))
    print(y)