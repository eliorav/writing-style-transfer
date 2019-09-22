import itertools

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from GAN.Decoder import Decoder
from GAN.Discriminator import Discriminator
from GAN.Encoder import Encoder
from GAN.Seq2Seq import Seq2Seq
from Utils.Services import init_weights, LambdaLRFn, load_model
from constants import dec_emb_dim, enc_emb_dim, g_hid_dim, g_n_layers, enc_dropout, dec_dropout, b1, b2, \
    MODEL_NAME_G_AB, MODEL_NAME_G_BA, MODEL_NAME_D_A, MODEL_NAME_D_B, d_emb_dim


def get_cycle_gan_network(
        g_input_dim,
        g_output_dim,
        device,
        pad_idx,
        sos_idx,
        should_load_pretrain_generators,
        should_load_pretrain_discriminators
):
    # Initialize generator and discriminator
    enc_a = Encoder(g_input_dim, enc_emb_dim, g_hid_dim, g_n_layers, enc_dropout, device)
    dec_a = Decoder(g_output_dim, dec_emb_dim, g_hid_dim, g_n_layers, dec_dropout, device)
    enc_b = Encoder(g_input_dim, enc_emb_dim, g_hid_dim, g_n_layers, enc_dropout, device)
    dec_b = Decoder(g_output_dim, dec_emb_dim, g_hid_dim, g_n_layers, dec_dropout, device)

    enc_a = enc_a.to(device)
    dec_a = dec_a.to(device)
    enc_b = enc_b.to(device)
    dec_b = dec_b.to(device)

    g_ab = Seq2Seq(enc_a, dec_a, sos_idx, device)
    g_ba = Seq2Seq(enc_b, dec_b, sos_idx, device)

    g_ab = g_ab.to(device)
    g_ba = g_ba.to(device)

    d_a = Discriminator(g_input_dim, d_emb_dim, pad_idx)
    d_b = Discriminator(g_input_dim, d_emb_dim, pad_idx)

    d_a = d_a.to(device)
    d_b = d_b.to(device)

    if should_load_pretrain_generators:
        # Load pretrained generators
        load_model(g_ab, MODEL_NAME_G_AB, device.type)
        load_model(g_ba, MODEL_NAME_G_BA, device.type)
    else:
        g_ab.apply(init_weights)
        g_ba.apply(init_weights)

    if should_load_pretrain_discriminators:
        # Load pretrained discriminators
        load_model(d_a, MODEL_NAME_D_A, device.type)
        load_model(d_b, MODEL_NAME_D_B, device.type)
    else:
        d_a.apply(init_weights)
        d_b.apply(init_weights)

    return g_ab, g_ba, d_a, d_b


def get_criterions(pad_idx, device):
    # Losses
    criterion_g_ab = nn.CrossEntropyLoss(ignore_index=pad_idx)
    criterion_g_ba = nn.CrossEntropyLoss(ignore_index=pad_idx)
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_discriminator = nn.BCEWithLogitsLoss()
    criterion_cycle = nn.CrossEntropyLoss(ignore_index=pad_idx)
    criterion_identity = nn.CrossEntropyLoss(ignore_index=pad_idx)

    criterion_gan = criterion_gan.to(device)
    criterion_discriminator = criterion_discriminator.to(device)
    criterion_cycle = criterion_cycle.to(device)
    criterion_identity = criterion_identity.to(device)

    return criterion_g_ab, criterion_g_ba, criterion_gan, criterion_discriminator, criterion_cycle, criterion_identity


def get_optimizers(g_ab, g_ba, d_a, d_b, lr):
    # Optimizers
    optimizer_g_ab = torch.optim.Adam(g_ab.parameters(), lr=lr, betas=(b1, b2))
    optimizer_g_ba = torch.optim.Adam(g_ba.parameters(), lr=lr, betas=(b1, b2))

    optimizer_g = torch.optim.Adam(
        itertools.chain(g_ab.parameters(), g_ba.parameters()), lr=lr, betas=(b1, b2)
    )
    optimizer_d_a = torch.optim.Adam(d_a.parameters(), lr=lr, betas=(b1, b2))
    optimizer_d_b = torch.optim.Adam(d_b.parameters(), lr=lr, betas=(b1, b2))
    return optimizer_g_ab, optimizer_g_ba, optimizer_g, optimizer_d_a, optimizer_d_b


def get_schedulers(optimizer_g, optimizer_d_a, optimizer_d_b, start_epoch, n_epochs, decay_epoch):
    # Learning rate update schedulers
    lr_scheduler_g = LambdaLR(
        optimizer_g, lr_lambda=LambdaLRFn(n_epochs, start_epoch, decay_epoch)
    )
    lr_scheduler_d_a = LambdaLR(
        optimizer_d_a, lr_lambda=LambdaLRFn(n_epochs, start_epoch, decay_epoch)
    )
    lr_scheduler_d_b = LambdaLR(
        optimizer_d_b, lr_lambda=LambdaLRFn(n_epochs, start_epoch, decay_epoch)
    )
    return lr_scheduler_g, lr_scheduler_d_a, lr_scheduler_d_b
