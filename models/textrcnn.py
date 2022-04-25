#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

class TextRCNN(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_layers,
                 dropout,
                 num_classes,
                 seq_len,
                 num_embeddings=None, embedding_dim=None,
                 embedding_matrix=None, trainalbe=True):
        super(TextRCNN, self).__init__()
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix,
                                                          freeze=trainalbe,
                                                          padding_idx=embedding_matrix.shape[0]-1)
        else:
            self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=num_embeddings-1)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.maxpool = nn.MaxPool1d(seq_len)
        self.embed = embedding_matrix.shape[1] if embedding_matrix else embedding_dim
        self.fc = nn.Linear(hidden_size * 2 + self.embed, num_classes)

    def forward(self, x):
        x, _ = x
        embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out

if __name__ == '__main__':
    net = TextRCNN(256, 2, 0.2, 15, 20, num_embeddings=10000, embedding_dim=300)

    # # need rm Embedding layer
    # from torchsummary import summary
    # summary(net, (20, 300))
    # exit(0)

    print(net)
    x = torch.randint(0, 10000, (1, 20))
    y = net((x, None))
    print(y)