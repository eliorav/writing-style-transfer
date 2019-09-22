import argparse
from DataProcessing.DataProcessor import DataProcessor

POSITIVE_REVIEW_STARS_LIMIT = 5
NEGATIVE_REVIEW_STARS_LIMIT = 1
NUM_OF_SENTENCES_LIMIT = 10
MIN_NUM_OF_WORDS_LIMIT = 2
MAX_NUM_OF_WORDS_LIMIT = 15
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive_review_stars_limit",
                        type=int,
                        default=POSITIVE_REVIEW_STARS_LIMIT,
                        help="limit for positive reviews"
                        )
    parser.add_argument("--negative_review_stars_limit",
                        type=int,
                        default=NEGATIVE_REVIEW_STARS_LIMIT,
                        help="limit for negative reviews"
                        )
    parser.add_argument("--num_of_sentences_limit",
                        type=int,
                        default=NUM_OF_SENTENCES_LIMIT,
                        help="limit for the number of sentences in the review"
                        )
    parser.add_argument("--min_num_of_words_limit",
                        type=int,
                        default=MIN_NUM_OF_WORDS_LIMIT,
                        help="limit for the minimum number of word in the review"
                        )
    parser.add_argument("--max_num_of_words_limit",
                        type=int,
                        default=MAX_NUM_OF_WORDS_LIMIT,
                        help="limit for the maximum number of word in the review"
                        )
    parser.add_argument("--test_size",
                        type=float,
                        default=TEST_SIZE,
                        help="test set size"
                        )
    parser.add_argument("--validation_size",
                        type=float,
                        default=VALIDATION_SIZE,
                        help="validation set size"
                        )
    parser.add_argument("--dataset_path",
                        type=str,
                        default=".data/yelp_academic_dataset_review.json",
                        help="path of the dataset"
                        )

    opt = parser.parse_args()

    data_processor = DataProcessor(
        opt.positive_review_stars_limit,
        opt.negative_review_stars_limit,
        opt.num_of_sentences_limit,
        opt.min_num_of_words_limit,
        opt.max_num_of_words_limit,
        opt.test_size,
        opt.validation_size,
    )
    data_processor.process_data(opt.dataset_path)


if __name__ == "__main__":
    main()
