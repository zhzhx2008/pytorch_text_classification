#conding=utf-8

import torch.nn as nn
import torch

batch_size=3
seq_len = 4
embedding_dim = 5
hidden_size = 6
num_layers = 2





# one lstm
input = torch.rand((3, 4, 5))
# print(input)
lstm = nn.LSTM(embedding_dim, hidden_size=hidden_size, batch_first=True, num_layers=2)
output, (h_n, c_n) = lstm(input)
print(output.size())
print(h_n.size())   # [num_layers, batch_size, hidden_size]
print(c_n.size())   # [num_layers, batch_size, hidden_size]
output_last = output[:, -1, :]
h_n_last = h_n[-1]
print(output_last.size())
print(h_n_last.size())
print(output_last.eq(h_n_last))





# two lstm
input = torch.rand((3, 4, 5))
# print(input)
lstm = nn.LSTM(embedding_dim, hidden_size=hidden_size, batch_first=True, num_layers=2, bidirectional=True)
output, (h_n, c_n) = lstm(input)
print(output.size())
print(h_n.size())   # [num_layers * 2, batch_size, hidden_size]
print(c_n.size())   # [num_layers * 2, batch_size, hidden_size]

# 获取反向的最后一个output
output_last = output[:, 0, -1 * hidden_size:]
# 获反向最后一层的h_n
h_n_last = h_n[-1]
print(output_last.size())
print(h_n_last.size())
# 反向最后的output等于最后一层的h_n
print(output_last.eq(h_n_last))

#获取正向的最后一个output
output_last = output[:, -1, :hidden_size]
#获取正向最后一层的h_n
h_n_last = h_n[-2]
print(output_last.size())
print(h_n_last.size())
print(output_last.eq(h_n_last))
