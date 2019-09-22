import random

import torch
from torch import nn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, sos_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.sos_idx = sos_idx

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, teacher_forcing_ratio = 0.5):
        """
        :param src: [src sent len, batch size]
        :param teacher_forcing_ratio is probability to use teacher forcing
        e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        :return:
        """
        batch_size = src.shape[1]
        max_len = src.shape[0]
        src_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, src_vocab_size).to(self.device)
        max_output = torch.zeros(max_len, batch_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        # first input_data to the decoder is the <sos> tokens
        input_data = src[0, :]
        max_output[0] = input_data

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input_data, hidden, cell)
            outputs[t] = output
            top = output.max(1)[1]
            max_output[t] = top

            teacher_force = random.random() < teacher_forcing_ratio
            input_data = src[t] if teacher_force else top


        return outputs, max_output.long()
