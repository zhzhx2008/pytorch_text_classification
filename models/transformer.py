#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class TransformerModel(nn.Module):
    def __init__(self, pad_size, dropout, device,
                 num_head, hidden,
                 num_encoder,
                 num_classes,
                 embedding_matrix=None, freeze=False,
                 num_embeddings=None, embedding_dim=None):
        super(TransformerModel, self).__init__()
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix,
                                                          freeze=freeze,
                                                          padding_idx=embedding_matrix.shape[0] - 1)
        else:
            self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=num_embeddings - 1)
        self.embedding_dim = embedding_dim
        if embedding_matrix is not None:
            self.embedding_dim = embedding_matrix.shape[1]
        self.position_embedding = Positional_Encoding(self.embedding_dim, pad_size=pad_size, dropout=dropout, device=device)
        self.encoder = Encoder(self.embedding_dim, num_head, hidden, dropout)
        self.encoders = nn.ModuleList(copy.deepcopy(self.encoder) for _ in range(num_encoder))
        self.fc1 = nn.Linear(pad_size * self.embedding_dim, num_classes)

    def forward(self, x):
        x, _ = x
        x = self.embedding(x)
        x = self.position_embedding(x)
        for encoder in self.encoders:
            x = encoder(x)
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        return out


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 0::1] = np.cos(self.pe[:, 0::1])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Positional_Wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        x = self.attention(x)
        x = self.feed_forward(x)
        return x


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        scale = K.size(-1) ** -0.5
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Positional_Wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout):
        super(Positional_Wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out


if __name__ == '__main__':
    net = TransformerModel(pad_size=20, dropout=0.5, device='cpu',
                           num_head=12, hidden=300,
                           num_encoder=12, num_classes=15,
                           num_embeddings=10000, embedding_dim=300)

    # # need rm Embedding layer
    # from torchinfo import summary
    # summary(net, (32, 20, 768))
    # exit(0)

    # # need rm Embedding layer
    # from torchsummary import summary
    # summary(net, (20, 768))
    # exit(0)

    print(net)
    x = torch.randint(0, 10000, (32, 20))
    y = net((x, None))
    print(y)