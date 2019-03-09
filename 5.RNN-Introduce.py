"""
  code by Tae Hwan Jung(Jeff Jung) @graykode
"""
import torch
import torch.nn as nn

batch = 1
seq_len = 5
input_size = 10
hidden_size= 3

rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)

# input of shape (batch, seq_len, input_size)
input = torch.rand([batch, seq_len, input_size])

# h_0 of shape (num_layers * num_directions, batch, hidden_size)
hidden = torch.rand([1 * 1, batch, hidden_size])

#  output of shape (batch, seq_len, num_directions * hidden_size)
#  final_hidden of shape (num_layers * num_directions, batch, hidden_size)
output, final_hidden = rnn(input, hidden)
print('final_hidden shape :', final_hidden.shape)
print(final_hidden)

# final_hidden state is same with last time seqeuence output
for index, out in enumerate(output[0]):
    print(index + 1, 'time seqeuence output :', out)