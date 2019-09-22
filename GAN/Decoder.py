from torch import nn


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, device):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.embedding = self.embedding.to(device)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.rnn = self.rnn.to(device)

        self.out = nn.Linear(hid_dim, output_dim)
        self.out = self.out.to(device)

        self.dropout = nn.Dropout(dropout)
        self.dropout = self.dropout.to(device)

    def forward(self, input_data, hidden, cell):
        """

        :param input_data: [batch size]
        :param hidden: [n layers, batch size, hid dim]
        :param cell: [n layers, batch size, hid dim]
        :return:
        prediction = [batch size, output dim]
        hidden = [n layers, batch size, hid dim]
        cell = [n layers, batch size, hid dim]
        """
        # input_data = [1, batch size]
        input_data = input_data.unsqueeze(0)

        # embedded = [1, batch size, emb dim]
        embedded = self.dropout(self.embedding(input_data))

        # output = [1, batch size, hid dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        prediction = self.out(output.squeeze(0))
        return prediction, hidden, cell
