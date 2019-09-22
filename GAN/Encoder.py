from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, device):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.embedding = self.embedding.to(device)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.rnn = self.rnn.to(device)

        self.dropout = nn.Dropout(dropout)
        self.dropout = self.dropout.to(device)

    def forward(self, src):
        """

        :param src: [src sent len, batch size]
        :return:
        hidden = [n layers * n directions, batch size, hid dim]
        cell = [n layers * n directions, batch size, hid dim]
        """
        embedded = self.dropout(self.embedding(src))

        # outputs are always from the top hidden layer
        # outputs = [src sent len, batch size, hid dim * n directions]
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell
