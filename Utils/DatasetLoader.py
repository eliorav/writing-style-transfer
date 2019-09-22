import spacy
from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset

from constants import DATA_FOLDER, POSITIVE_FILE_EXTENSION, NEGATIVE_FILE_EXTENSION

spacy_en = spacy.load('en')


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


def load_dataset(batch_size, device):
    """
    Load the dataset from the files into iterator and initialize the vocabulary
    :param batch_size
    :param device
    :return: source and data iterators
    """
    source = Field(tokenize=tokenize_en,
                   init_token='<sos>',
                   eos_token='<eos>',
                   lower=True)

    train_data, valid_data, test_data = TranslationDataset.splits(
        path=DATA_FOLDER,
        exts=(POSITIVE_FILE_EXTENSION, NEGATIVE_FILE_EXTENSION),
        fields=(source, source)
    )
    source.build_vocab(train_data, min_freq=5)
    return source, BucketIterator.splits(
        (train_data, valid_data, test_data),
        shuffle=True,
        batch_size=batch_size,
        device=device)
