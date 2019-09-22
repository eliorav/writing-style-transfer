import time
import torch
import math
from tqdm import tqdm

from Utils.Services import print_message, get_time, get_bleu_score, save_model, save_stats
from constants import MODEL_NAME_G_AB, MODEL_NAME_G_BA, MODEL_NAME_D_A, MODEL_NAME_D_B


def evaluate_generators_epoch(
        g_ab,
        g_ba,
        data_iterator,
        criterion_g_ab,
        criterion_g_ba,
        device
):
    """
    Evaluate the generators
    :param g_ab:
    :param g_ba:
    :param data_iterator:
    :param criterion_g_ab:
    :param criterion_g_ba:
    :param device:
    """
    g_ab.eval()
    g_ba.eval()

    g_ab_epoch_loss = 0
    g_ba_epoch_loss = 0

    print_message(f'Start Evaluation')
    with torch.no_grad():
        with tqdm(total=len(data_iterator)) as pbar:
            for i, batch in tqdm(enumerate(data_iterator)):
                pbar.update(1)
                torch.cuda.empty_cache()

                # Set model input
                real_a = batch.src.to(device)
                real_b = batch.trg.to(device)

                # ------------------
                #  Train G_AB
                # ------------------
                torch.cuda.empty_cache()

                identity_out_ab, _ = g_ab(real_b, 0)
                identity_out_ab = identity_out_ab[1:].view(-1, identity_out_ab.shape[-1])
                g_ab_epoch_loss += criterion_g_ab(identity_out_ab, real_b[1:].view(-1)).item()

                # ------------------
                #  Train G_AB
                # ------------------
                torch.cuda.empty_cache()

                identity_out_ba, _ = g_ba(real_a, 0)
                identity_out_ba = identity_out_ba[1:].view(-1, identity_out_ba.shape[-1])
                g_ba_epoch_loss += criterion_g_ba(identity_out_ba, real_a[1:].view(-1)).item()

        return g_ab_epoch_loss / len(data_iterator), g_ba_epoch_loss / len(data_iterator)


def train_generators_epoch(
        clip,
        g_ab,
        g_ba,
        data_iterator,
        criterion_g_ab,
        criterion_g_ba,
        optimizer_g_ab,
        optimizer_g_ba,
        device
):
    """
    Train one epoch of the generators
    """
    g_ab.train()
    g_ba.train()

    g_ab_epoch_loss = 0
    g_ba_epoch_loss = 0

    print_message(f'Start Training')
    with tqdm(total=len(data_iterator)) as pbar:
        for i, batch in enumerate(data_iterator):
            pbar.update(1)
            torch.cuda.empty_cache()

            # Set model input
            real_a = batch.src.to(device)
            real_b = batch.trg.to(device)

            # ------------------
            #  Train G_AB
            # ------------------
            torch.cuda.empty_cache()
            optimizer_g_ab.zero_grad()

            identity_out_ab, _ = g_ab(real_b)
            identity_out_ab = identity_out_ab[1:].view(-1, identity_out_ab.shape[-1])
            loss_id_b = criterion_g_ab(identity_out_ab, real_b[1:].view(-1))

            loss_id_b.backward()
            torch.nn.utils.clip_grad_norm_(g_ab.parameters(), clip)
            optimizer_g_ab.step()

            g_ab_epoch_loss += loss_id_b.item()

            # ------------------
            #  Train G_AB
            # ------------------
            torch.cuda.empty_cache()
            optimizer_g_ba.zero_grad()

            identity_out_ba, _ = g_ba(real_a)
            identity_out_ba = identity_out_ba[1:].view(-1, identity_out_ba.shape[-1])
            loss_id_a = criterion_g_ba(identity_out_ba, real_a[1:].view(-1))

            loss_id_a.backward()
            torch.nn.utils.clip_grad_norm_(g_ba.parameters(), clip)
            optimizer_g_ba.step()

            g_ba_epoch_loss += loss_id_a.item()

    return g_ab_epoch_loss / len(data_iterator), g_ba_epoch_loss / len(data_iterator)


def train_generators(
        n_epochs,
        clip,
        g_ab,
        g_ba,
        data_iterators,
        criterion_g_ab,
        criterion_g_ba,
        optimizer_g_ab,
        optimizer_g_ba,
        device
):
    """
    Train only the generators
    :param n_epochs:
    :param clip:
    :param g_ab:
    :param g_ba:
    :param data_iterators:
    :param criterion_g_ab:
    :param criterion_g_ba:
    :param optimizer_g_ab:
    :param optimizer_g_ba:
    :param device:
    """
    print_message(f'\n\nStart Generators Training')
    best_valid_loss_ab = float('inf')
    best_valid_loss_ba = float('inf')

    for epoch in range(n_epochs):
        print_message(f'[Epoch {epoch + 1}/{n_epochs}]')
        start_time = time.time()
        train_loss_ab, train_loss_ba = train_generators_epoch(clip,
                                                              g_ab,
                                                              g_ba,
                                                              data_iterators[0],
                                                              criterion_g_ab,
                                                              criterion_g_ba,
                                                              optimizer_g_ab,
                                                              optimizer_g_ba,
                                                              device)
        valid_loss_ab, valid_loss_ba = evaluate_generators_epoch(
            g_ab,
            g_ba,
            data_iterators[1],
            criterion_g_ab,
            criterion_g_ba,
            device
        )

        end_time = time.time()
        epoch_mins, epoch_secs = get_time(start_time, end_time)

        if valid_loss_ab < best_valid_loss_ab:
            best_valid_loss_ab = valid_loss_ab
            save_model(g_ab, MODEL_NAME_G_AB)

        if valid_loss_ba < best_valid_loss_ba:
            best_valid_loss_ba = valid_loss_ba
            save_model(g_ba, MODEL_NAME_G_BA)

        print_message(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print_message(f'\tTrain Loss AB: {train_loss_ab:.3f} | Train PPL: {math.exp(train_loss_ab):7.3f}')
        print_message(f'\tTrain Loss BA: {train_loss_ba:.3f} | Train PPL: {math.exp(train_loss_ba):7.3f}')
        print_message(f'\tVal. Loss AB: {valid_loss_ab:.3f} | Val. PPL: {math.exp(valid_loss_ab):7.3f}')
        print_message(f'\tVal. Loss BA: {valid_loss_ba:.3f} | Val. PPL: {math.exp(valid_loss_ba):7.3f}')


def train_cycle_gan_epoch(
        device,
        g_ab,
        g_ba,
        d_a,
        d_b,
        data_iterator,
        optimizer_g,
        optimizer_d_a,
        optimizer_d_b,
        criterion_identity,
        criterion_gan,
        criterion_cycle,
        criterion_discriminator,
        lambda_cyc,
        lambda_id,
        lambda_adv,
):
    """
    Train one epoch of cycle GAN
    :param device: cuda or cpu
    :param g_ab: generator from domain a to domain b
    :param g_ba: generator from domain b to domain a
    :param d_a: discriminator for domain a
    :param d_b: discriminator for domain b
    :param data_iterator:
    :param optimizer_g:
    :param optimizer_d_a:
    :param optimizer_d_b:
    :param criterion_identity:
    :param criterion_gan:
    :param criterion_cycle:
    :param criterion_discriminator:
    :param lambda_id: identity loss weight
    :param lambda_cyc: cycle loss weight
    """
    g_ab.train()
    g_ba.train()
    d_a.train()
    d_b.train()

    with tqdm(total=len(data_iterator)) as pbar:
        for i, batch in enumerate(data_iterator):
            pbar.update(1)
            torch.cuda.empty_cache()
            batch_size = len(batch)
            # Set model input
            real_a = batch.src.to(device)
            real_b = batch.trg.to(device)

            # Adversarial ground truths
            valid = torch.ones(batch_size, requires_grad=False).to(device)
            fake = torch.zeros(batch_size, requires_grad=False).to(device)

            # ------------------
            #  Train Generators
            # ------------------
            optimizer_g.zero_grad()

            # Identity loss
            torch.cuda.empty_cache()
            identity_out_ba, _ = g_ba(real_a)
            identity_out_ba = identity_out_ba[1:].view(-1, identity_out_ba.shape[-1])
            loss_id_a = criterion_identity(identity_out_ba, real_a[1:].view(-1))

            identity_out_ab, _ = g_ab(real_b)
            identity_out_ab = identity_out_ab[1:].view(-1, identity_out_ab.shape[-1])
            loss_id_b = criterion_identity(identity_out_ab, real_b[1:].view(-1))

            loss_identity = (loss_id_a.item() + loss_id_b.item()) / 2

            # GAN loss
            torch.cuda.empty_cache()
            _, fake_b = g_ab(real_a)
            loss_gan_ab = criterion_gan(d_b(fake_b).squeeze(1), valid)

            _, fake_a = g_ba(real_b)
            loss_gan_ba = criterion_gan(d_a(fake_a).squeeze(1), valid)

            loss_gan = (loss_gan_ab.item() + loss_gan_ba.item()) / 2

            # Cycle loss
            torch.cuda.empty_cache()
            recov_a, _ = g_ba(fake_b)
            recov_a = recov_a[1:].view(-1, recov_a.shape[-1])
            loss_cycle_a = criterion_cycle(recov_a, real_a[1:].view(-1))
            recov_b, _ = g_ab(fake_a)
            recov_b = recov_b[1:].view(-1, recov_b.shape[-1])
            loss_cycle_b = criterion_cycle(recov_b, real_b[1:].view(-1))

            loss_cycle = (loss_cycle_a + loss_cycle_b) / 2

            # total loss
            loss_g = lambda_adv * loss_gan + lambda_cyc * loss_cycle + lambda_id * loss_identity

            loss_g.backward()
            optimizer_g.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------
            torch.cuda.empty_cache()
            optimizer_d_a.zero_grad()

            # Real loss
            loss_real = criterion_discriminator(d_a(real_a).squeeze(1), valid)
            loss_fake = criterion_discriminator(d_a(fake_a).squeeze(1), fake)
            # total loss
            loss_d_a = (loss_real + loss_fake) / 2

            loss_d_a.backward()
            optimizer_d_a.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------
            torch.cuda.empty_cache()
            optimizer_d_b.zero_grad()

            # real loss
            loss_real = criterion_discriminator(d_b(real_b).squeeze(1), valid)
            loss_fake = criterion_discriminator(d_b(fake_b).squeeze(1), fake)
            # total loss
            loss_d_b = (loss_real + loss_fake) / 2

            loss_d_b.backward()
            optimizer_d_b.step()

            loss_d = (loss_d_a.item() + loss_d_b.item()) / 2

            if i == len(data_iterator) - 1:
                print_message(
                    f'[Batch {i + 1}/{len(data_iterator)}] [D loss: {loss_d}] [G loss: {loss_g.item()}, adv: {loss_gan}, cycle: {loss_cycle}, identity: {loss_identity}]')


def evaluate_cycle_gan(
        source,
        device,
        g_ab,
        g_ba,
        d_a,
        d_b,
        data_iterator,
        criterion_gan
):
    """
    Evaluate the cycle GAN architecture
    :param source:
    :param device:
    :param g_ab:
    :param g_ba:
    :param d_a:
    :param d_b:
    :param data_iterator:
    :param criterion_gan:
    """
    # the Generators should keep the dropout for evaluation
    d_a.eval()
    d_b.eval()

    loss_gan_ab = 0
    loss_gan_ba = 0
    bleu_score_a = 0
    bleu_score_b = 0

    with torch.no_grad():
        data_iterator_length = len(data_iterator)
        with tqdm(total=len(data_iterator)) as pbar:
            for i, batch in enumerate(data_iterator):
                pbar.update(1)
                batch_size = len(batch)
                # Set model input
                real_a = batch.src.to(device)
                real_b = batch.trg.to(device)

                # Adversarial ground truths
                valid = torch.ones(batch_size, requires_grad=False).to(device)

                # GAN loss
                _, fake_b = g_ab(real_a, 0)
                loss_gan_ab += criterion_gan(d_b(fake_b).squeeze(1), valid).item()

                _, fake_a = g_ba(real_b, 0)
                loss_gan_ba += criterion_gan(d_a(fake_a).squeeze(1), valid).item()

                bleu_score_a += get_bleu_score(source, real_a, fake_b)
                bleu_score_b += get_bleu_score(source, real_b, fake_a)

        loss_gan_ab /= data_iterator_length
        loss_gan_ba /= data_iterator_length
        bleu_score_a /= data_iterator_length
        bleu_score_b /= data_iterator_length

    return float(loss_gan_ab), float(loss_gan_ba), bleu_score_a, bleu_score_b


def train_cycle(
        start_epoch,
        n_epochs,
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
        lambda_cyc,
        lambda_id,
        lambda_adv,
        lr_scheduler_g,
        lr_scheduler_d_a,
        lr_scheduler_d_b,
):
    """
    Train the networks using cycle GAN architecture
    :param start_epoch:
    :param n_epochs:
    :param source:
    :param device:
    :param g_ab:
    :param g_ba:
    :param d_a:
    :param d_b:
    :param iterators:
    :param optimizer_g:
    :param optimizer_d_a:
    :param optimizer_d_b:
    :param criterion_identity:
    :param criterion_gan:
    :param criterion_cycle:
    :param criterion_discriminator:
    :param lambda_cyc:
    :param lambda_id:
    :param lambda_adv:
    :param lr_scheduler_g:
    :param lr_scheduler_d_a:
    :param lr_scheduler_d_b:
    """
    best_loss_ab = float('inf')
    best_loss_ba = float('inf')
    for epoch in range(start_epoch, n_epochs):
        print_message(f'[Epoch {epoch + 1}/{n_epochs}]')
        start_time = time.time()

        print_message(f'Training the network')
        train_cycle_gan_epoch(
            device,
            g_ab,
            g_ba,
            d_a,
            d_b,
            iterators[0],
            optimizer_g,
            optimizer_d_a,
            optimizer_d_b,
            criterion_identity,
            criterion_gan,
            criterion_cycle,
            criterion_discriminator,
            lambda_cyc,
            lambda_id,
            lambda_adv
        )

        print_message(f'Evaluate the network')
        loss_gan_ab, loss_gan_ba, bleu_score_a, bleu_score_b = evaluate_cycle_gan(
            source,
            device,
            g_ab,
            g_ba,
            d_a,
            d_b,
            iterators[1],
            criterion_gan
        )

        # Update learning rates
        lr_scheduler_g.step()
        lr_scheduler_d_a.step()
        lr_scheduler_d_b.step()

        print_message(
            f'loss_gan_ab: {loss_gan_ab} | loss_gan_ba: {loss_gan_ba} | bleu_score_a: {bleu_score_a} | bleu_score_b: {bleu_score_b}')
        save_stats(loss_gan_ab, loss_gan_ba, bleu_score_a, bleu_score_b)

        end_time = time.time()
        epoch_mins, epoch_secs = get_time(start_time, end_time)
        print_message(f'Epoch: {(epoch + 1):02} | Time: {epoch_mins}m {epoch_secs}s \n\n')

        # Save models
        save_model(d_a, MODEL_NAME_D_A)
        save_model(d_b, MODEL_NAME_D_B)

        if loss_gan_ab < best_loss_ab:
            best_loss_ab = loss_gan_ab
            save_model(g_ab, MODEL_NAME_G_AB)

        if loss_gan_ba < best_loss_ba:
            best_loss_ba = loss_gan_ba
            save_model(g_ba, MODEL_NAME_G_BA)
