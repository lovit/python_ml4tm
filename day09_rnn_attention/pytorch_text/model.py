import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
        bias=True, batch_first=True, dropout=0., bidirectional=False,
        mode='gru', embedding=None, num_vocab=-1):

        super().__init__()

        mode = mode.lower()
        assert mode in ['rnn', 'gru', 'lstm']

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.is_lstm = mode == 'lstm'
        self.num_directions = 1 + bidirectional

        # prepare embedding layer
        if (embedding is None) and (num_vocab > 0):
            embedding = nn.Embedding(num_vocab+1, input_size, padding_idx=num_vocab)
        elif embedding is not None:
            assert embedding.weight.size()[1] == input_size
        self.embedding = embedding

        if mode == 'gru':
            self.layer = nn.GRU(input_size, hidden_size, num_layers,
                bias, batch_first, dropout, bidirectional)
        elif mode == 'lstm':
            self.layer = nn.LSTM(input_size, hidden_size, num_layers,
                bias, batch_first, dropout, bidirectional)
        else:
            self.layer = nn.RNN(input_size, hidden_size, num_layers,
                bias, batch_first, dropout, bidirectional)

        self.h2o = nn.Linear(self.num_directions * hidden_size, output_size)

    def forward(self, input, hidden=None):
        """
        Assume that input is batch_first; shape = (batch size, sequence length, input size)
        or PackedSequence.

        If input type is LongTensor, use embedding layer to transform
        integer sequence to vector sequence.
        """
        if isinstance(input[0], torch.LongTensor):
            if isinstance(input, PackedSequence):
                inp_sequence, lengths, sorted_indices, unsorted_indices = input
                inp_sequence = self.embedding(inp_sequence)
                input = PackedSequence(inp_sequence, lengths, sorted_indices, unsorted_indices)
            else:
                input = self.embedding(input)

        out, hn = self.layer(input, hidden)

        if self.is_lstm:
            # (hidden, cell)
            hn, cn = hn

        if isinstance(out, PackedSequence):
            out, out_lengths, _, _ = out

        batch_size = hn.size()[1]
        # (num direction * num layers, batch size, hidden size)
        #   -> (num layers, num directions, batch size, hidden size)
        #   -> (num direction, batch size, hidden size)
        hn = hn.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)[-1]

        #   -> (batch size, num direction, hidden size)
        hn = hn.permute([1, 0, 2])

        #   -> (batch size, num_direction * hidden size)
        hn = torch.flatten(hn, start_dim=1)

        # hidden in last layer to output (log likelihood)
        # (batch size, output size)
        ll = F.log_softmax(self.h2o(hn), dim=-1)

        return ll

    def init_hidden(self, batch_size, device):
        # (sum of all layers, batch size, hidden size)
        zeros = torch.zeros(self.num_layers * self.num_directions,
            batch_size, hidden_size, device=device)
        if self.is_lstm:
            return (zeros, zeros)
        else:
            return zeros
