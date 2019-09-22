import torch
from torch import nn
import logging

from constants import MODELS_FOLDER
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import csv

logging.basicConfig(filename='app.log', level=logging.DEBUG)


def LambdaLRFn(n_epochs, offset, decay_start_epoch):
    """
    :param n_epochs
    :param offset
    :param decay_start_epoch
    """
    return lambda epoch: 1.0 - max(0, epoch + offset - decay_start_epoch) / (
            n_epochs - decay_start_epoch)


def init_weights(m):
    """
    Init the network's weights uniformly
    :param m
    """
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def get_time(start_time, end_time):
    """
    Get duration time in minutes and seconds
    :param start_time
    :param end_time
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def print_message(message):
    """
    Print the log message
    :param message
    """
    logging.error(message)
    print(message)


def get_model_path_by_epoch(model_name):
    """
    Get the model page by modal name
    :param model_name
    """
    return f"{MODELS_FOLDER}/{model_name}.pth"


def save_model(g, modal_name):
    """
    Save the model
    :param g
    :param modal_name
    """
    torch.save(g.state_dict(), get_model_path_by_epoch(modal_name))


def load_model(model, name, device_type):
    """
    Load the model
    :param model
    :param name
    :param device_type: cuda or cpu
    """
    model.load_state_dict(torch.load(get_model_path_by_epoch(name), map_location=device_type))


def get_sentence_from_tensor(source, tensor):
    """
    Convert sentence tensor into textual sentence
    :param source
    :param tensor
    """
    items = tensor.transpose(0, 1)
    res = [[source.vocab.itos[ind] for ind in ids] for ids in items]
    return [text[1: text.index('<eos>')] if '<eos>' in text else text[1:] for text in res]


def get_bleu_score(source, real, fake):
    """
    Return average bleu score for the batch between the original sentences to the transformed ones
    :param source
    :param real
    :param fake
    """
    smoother = SmoothingFunction()
    real_text = get_sentence_from_tensor(source, real)
    fake_text = get_sentence_from_tensor(source, fake)
    return np.mean([sentence_bleu([real_text[i]], fake_text[i], smoothing_function=smoother.method4) * 100 for i in
                    range(len(real_text))])


def save_stats(loss_gan_ab, loss_gan_ba, bleu_score_a, bleu_score_b):
    """
    Save statistics
    :param loss_gan_ab
    :param loss_gan_ba
    :param bleu_score_a
    :param bleu_score_b
    """
    with open('./stats.csv', mode='a') as status_file:
        status_writer = csv.writer(status_file)
        status_writer.writerow([loss_gan_ab, loss_gan_ba, bleu_score_a, bleu_score_b])
