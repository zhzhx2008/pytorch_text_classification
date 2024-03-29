#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


'''Recurrent Neural Network for Text Classification with Multi-Task Learning'''


class TextRNNModel(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_layers,
                 dropout,
                 num_classes,
                 num_embeddings=None, embedding_dim=None, padding_idx=False,
                 embedding_matrix=None, freeze=False):
        super(TextRNNModel, self).__init__()
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix,
                                                          freeze=freeze,
                                                          padding_idx=padding_idx)
        else:
            self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.embedding_dim = embedding_matrix.shape[1] if embedding_matrix is not None else embedding_dim
        self.lstm = nn.LSTM(self.embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x, seq_lens = x
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

    # def forward(self, x):
    #     '''变长RNN'''
    #     out, seq_lens = x
    #     tensor_in = self.embedding(out)
    #     _, idx_sort = torch.sort(seq_lens, dim=0, descending=True)
    #     _, idx_unsort = torch.sort(idx_sort, dim=0)
    #     # order_seq_lengths = list(seq_lens[idx_sort])
    #     order_seq_lengths = torch.index_select(seq_lens, dim=0, index=idx_sort)
    #     order_tensor_in = torch.index_select(tensor_in, dim=0, index=idx_sort)
    #     x_packed = nn.utils.rnn.pack_padded_sequence(order_tensor_in, order_seq_lengths.tolist(), batch_first=True)  # lengths: (Tensor or list(int)), (must be on the CPU if provided as a tensor).
    #     y_packed, (h_n, c_n) = self.lstm(x_packed)
    #     # y_sort, length = nn.utils.rnn.pad_packed_sequence(y_packed, batch_first=True)
    #     # y = torch.index_select(y_sort, dim=0, index=idx_unsort)
    #     last_h = torch.index_select(torch.cat((h_n[-2], h_n[-1]), -1), dim=0, index=idx_unsort)
    #     out = self.fc(last_h)
    #     return out



if __name__ == '__main__':
    # tensor_in = torch.FloatTensor([[1, 0, 0], [1, 2, 3]]).resize_(2, 3, 1)
    # seq_lens = torch.IntTensor([1, 3])
    # _, idx_sort = torch.sort(seq_lens, dim=0, descending=True)
    # _, idx_unsort = torch.sort(idx_sort, dim=0)
    # order_seq_lengths = torch.index_select(seq_lens, dim=0, index=idx_sort)
    # order_tensor_in = torch.index_select(tensor_in, dim=0, index=idx_sort)
    # x_packed = nn.utils.rnn.pack_padded_sequence(order_tensor_in, order_seq_lengths, batch_first=True)
    # # rnn = nn.RNN(1, 2, 1, batch_first=True)
    # # y_packed, h_n = rnn(x_packed)
    # lstm = nn.LSTM(1, 2, 2, batch_first=True, bidirectional=True)
    # # lstm = nn.LSTM(1, 2, 2, batch_first=True)
    # y_packed, (h_n, c_n) = lstm(x_packed)
    # y_sort, length = nn.utils.rnn.pad_packed_sequence(y_packed, batch_first=True)
    # y = torch.index_select(y_sort, dim=0, index=idx_unsort)
    # last_h_1 = torch.index_select(h_n[-1], dim=0, index=idx_unsort)
    # last_h_2 = torch.index_select(h_n[-2], dim=0, index=idx_unsort)
    # last_h = torch.index_select(torch.cat((h_n[-2], h_n[-1]), -1), dim=0, index=idx_unsort)
    # print()
    # exit(0)



    net = TextRNNModel(256, 2, 0.2, 15, num_embeddings=10000, embedding_dim=300)

    # # need rm Embedding layer
    # from torchinfo import summary
    # summary(net, (32, 20, 300))
    # exit(0)

    print(net)
    x = torch.randint(0, 10000, (5, 20))
    seq_lens = torch.tensor([3,5,2,1,4])
    y = net((x, seq_lens))
    print(y)