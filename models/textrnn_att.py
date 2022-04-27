#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


'''Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification'''


class TextRNNAttModel(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_layers,
                 dropout,
                 num_classes,
                 num_embeddings=None, embedding_dim=None,
                 embedding_matrix=None, freeze=False):
        super(TextRNNAttModel, self).__init__()
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix,
                                                          freeze=freeze,
                                                          padding_idx=embedding_matrix.shape[0] - 1)
        else:
            self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=num_embeddings - 1)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(hidden_size * 2))
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x, seq_lens = x
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        alpha = F.softmax(torch.matmul(self.tanh1(lstm_out), self.w), dim=1).unsqueeze(-1)
        out = lstm_out * alpha
        out = torch.sum(out, 1)
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    net = TextRNNAttModel(256, 2, 0.2, 15, num_embeddings=10000, embedding_dim=300)

    # # need rm Embedding layer
    # from torchinfo import summary
    # summary(net, (32, 20, 300))
    # exit(0)

    print(net)
    x = torch.randint(0, 10000, (32, 20))
    y = net((x, None))
    print(y)