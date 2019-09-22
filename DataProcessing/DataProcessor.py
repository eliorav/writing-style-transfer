import codecs
import json
import os
import random
import re
import shutil

from tqdm import tqdm

from constants import DATA_FOLDER, TEST_TYPE, VALIDATION_TYPE, TRAIN_TYPE, POSITIVE_FILE_EXTENSION, \
    NEGATIVE_FILE_EXTENSION, DATASET_TYPES

SPLIT_SENTENCES_REGEX = '(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s' # split sentences from a paragraph

STARS_FIELD = 'stars'
TEXT_FIELD = 'text'


class DataProcessor:
    @staticmethod
    def init_files():
        """
        Initialize the data folder
        """
        if os.path.exists(DATA_FOLDER):
            shutil.rmtree(DATA_FOLDER)
        os.makedirs(DATA_FOLDER)

    @staticmethod
    def process_sentences(sentences, output_path):
        """
        Save the sentences in the expected file type
        :param sentences
        :param output_path
        """
        sentences = DataProcessor.clean_sentences(sentences)
        with open(output_path, 'a+') as out:
            for sentence in sentences:
                out.write(sentence)
                out.write("\n")

    @staticmethod
    def clean_sentences(sentences):
        """
        Clean sentences
        :param sentences
        """
        return [
            sentence
                .strip()
                .replace('\r\n', '')
                .replace('\r', '')
                .replace('\n', '')
                .encode('ascii', 'ignore')
                .decode()
                .lower()
            for sentence in sentences
        ]

    @staticmethod
    def clean_data(sentences):
        return list(map(lambda sentence: sentence.strip(), sentences))

    def __init__(
            self,
            positive_review_stars_limit,
            negative_review_stars_limit,
            num_of_sentences_limit,
            min_num_of_words_limit,
            max_num_of_words_limit,
            test_size,
            validation_size
    ):
        self.positive_review_stars_limit = positive_review_stars_limit
        self.negative_review_stars_limit = negative_review_stars_limit
        self.num_of_sentences_limit = num_of_sentences_limit
        self.min_num_of_words_limit = min_num_of_words_limit
        self.max_num_of_words_limit = max_num_of_words_limit
        self.dataset_random_array = []
        self.init_dataset_random_array(test_size, validation_size)

    def init_dataset_random_array(self, test_size, validation_size):
        test_items = int(test_size * 100)
        validation_items = int(validation_size * 100)
        self.dataset_random_array = test_items * [TEST_TYPE] + \
                                    validation_items * [VALIDATION_TYPE] + \
                                    (100 - test_items - validation_items) * [TRAIN_TYPE]

    def process_data(self, path):
        """
        Process the dataset
        1. split the reviews into sentences
        2. filter out reviews with more then self.num_of_sentences_limit sentences
        3. filter the sentences that exceed self.max_num_of_words_limit words and
        less than self.min_num_of_words_limit words
        4. classify sentences that belong to review with more then self.positive_review_stars_limit as positive
        5. classify sentences that belong to review with less then self.process_negative_sentences as negative
        :param path
        """
        DataProcessor.init_files()
        with codecs.open(path, 'r', 'utf-8') as data_file:
            for line in tqdm(data_file):
                try:
                    review = json.loads(line)
                    sentences = DataProcessor.clean_data(re.split(SPLIT_SENTENCES_REGEX, review[TEXT_FIELD]))

                    # we filter out reviews that exceed 10 sentences
                    if len(sentences) > self.num_of_sentences_limit:
                        continue

                    # We filter the sentences that exceed self.max_num_of_words_limit words and
                    # less than self.min_num_of_words_limit words
                    sentences = [sentence for sentence in sentences if
                                 self.max_num_of_words_limit > len(sentence.split()) > self.min_num_of_words_limit]

                    if review[STARS_FIELD] >= self.positive_review_stars_limit:
                        self.process_positive_sentences(sentences)
                    elif review[STARS_FIELD] <= self.negative_review_stars_limit:
                        self.process_negative_sentences(sentences)
                except Exception as e:  # non unicode chars
                    pass

    def process_positive_sentences(self, sentences):
        DataProcessor.process_sentences(sentences, output_path=self.get_random_dataset_path(POSITIVE_FILE_EXTENSION))

    def process_negative_sentences(self, sentences):
        DataProcessor.process_sentences(sentences, output_path=self.get_random_dataset_path(NEGATIVE_FILE_EXTENSION))

    def get_random_dataset_path(self, extension):
        """
        Get the sentence file type (train, val, test) randomly
        :param extension
        """
        dataset_type = self.dataset_random_array[random.randint(0, 99)]
        return "{}/{}{}".format(DATA_FOLDER, DATASET_TYPES[dataset_type], extension)
