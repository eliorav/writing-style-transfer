import torch
import argparse

from GAN.CycleGan import get_cycle_gan_network, get_criterions, get_optimizers, get_schedulers
from constants import g_clip
from GAN.train import train_generators, train_cycle
from Utils.DatasetLoader import load_dataset

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--start_epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs for cycle GAN training")
    parser.add_argument("--g_n_epochs", type=int, default=10, help="number of epochs for generator training")
    parser.add_argument("--decay_epoch", type=int, default=25, help="epoch from which to start lr decay")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--lambda_cyc", type=float, default=8.0, help="cycle loss weight")
    parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
    parser.add_argument("--lambda_adv", type=float, default=1.0, help="generator adversarial loss weight")
    parser.add_argument("--should_pretrain_generators", type=bool, default=True, help="should pre train the generators")
    parser.add_argument("--should_load_pretrain_generators", type=bool, default=False,
                        help="should load the pre train generators")
    parser.add_argument("--should_load_pretrain_discriminators", type=bool, default=False,
                        help="should load the pre train discriminators")
    opt = parser.parse_args()

    # Load the Dataset
    source, iterators = load_dataset(opt.batch_size, device)

    G_INPUT_DIM = len(source.vocab)
    G_OUTPUT_DIM = len(source.vocab)

    SOS_IDX = source.vocab.stoi['<sos>']
    PAD_IDX = source.vocab.stoi['<pad>']

    # Losses
    criterion_g_ab, criterion_g_ba, criterion_gan, criterion_discriminator, criterion_cycle, criterion_identity = get_criterions(
        PAD_IDX, device)

    # Initialize generator and discriminator
    g_ab, g_ba, d_a, d_b = get_cycle_gan_network(G_INPUT_DIM, G_OUTPUT_DIM, device, PAD_IDX, SOS_IDX,
                                                 opt.should_load_pretrain_generators,
                                                 opt.should_load_pretrain_discriminators)

    # Optimizers
    optimizer_g_ab, optimizer_g_ba, optimizer_g, optimizer_d_a, optimizer_d_b = get_optimizers(g_ab, g_ba, d_a, d_b,
                                                                                               opt.lr)

    # Learning rate update schedulers
    lr_scheduler_g, lr_scheduler_d_a, lr_scheduler_d_b = get_schedulers(optimizer_g, optimizer_d_a, optimizer_d_b,
                                                                        opt.start_epoch, opt.n_epochs, opt.decay_epoch)
    if opt.should_pretrain_generators:
        train_generators(
            opt.g_n_epochs,
            g_clip,
            g_ab,
            g_ba,
            iterators,
            criterion_g_ab,
            criterion_g_ba,
            optimizer_g_ab,
            optimizer_g_ba,
            device
        )

    train_cycle(
        opt.start_epoch,
        opt.n_epochs,
        source,
        device,
        g_ab,
        g_ba,
        d_a,
        d_b,
        iterators,
        optimizer_g,
        optimizer_d_a,
        optimizer_d_b,
        criterion_identity,
        criterion_gan,
        criterion_cycle,
        criterion_discriminator,
        opt.lambda_cyc,
        opt.lambda_id,
        opt.lambda_adv,
        lr_scheduler_g,
        lr_scheduler_d_a,
        lr_scheduler_d_b
    )


if __name__ == "__main__":
    main()
