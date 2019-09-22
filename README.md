<p align="center">
  <h3 align="center">Writing Style Transfer</h3>

  <p align="center">
    <a href="https://github.com/eliorav/writing-style-transfer/blob/master/writing_style_transfer_paper.pdf"><strong>Read Our Paper</strong></a>
  </p>
</p>

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Build The Dataset](#build-the-dataset)
  * [Train The Network](#train-the-network)
  * [See The Results](#see-the-results)
* [Hacks and Observations](#hacks-and-observations)
* [References](#references)
  * [Papers](#papers)
  * [Other Resources](#other-resources)
* [Contact](#contact)

<!-- ABOUT THE PROJECT -->
## About The Project

This project focuses on writing style transfer based on non-parallel text.
Our task focus on changing the text style, with the challenge of keeping the original content of the text as possible.
We propose a sequence to sequence model for learning to transform a text from a source domain to a different target domain in the absence of paired examples.
Our goal is to learn the relationship of two given domains and creates two functions that can convert each domain to its counterpart to perform writing style transfer.
We used a cross-domain writing style transformation using Cycle GAN.

<!-- BUILT WITH -->
### Built With
* [Python 3.6 +](https://www.python.org/)
* [numpy](https://jquery.com)
* [nltk](https://www.nltk.org/)
* [spacy](https://spacy.io/)
* [pytorch](https://pytorch.org/)
* [torchtext](https://torchtext.readthedocs.io/en/latest/)
* [pandas](https://pandas.pydata.org/)
* [plotly](https://plot.ly/python/)

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

Install requirements libraries

1. Install python libraries
```sh
pip install --user -r requirements.txt
```

2. Download spacy english model
```sh
python -m spacy download --user en
```

### Build The Dataset

1. Download the Yelp Dataset from [https://www.kaggle.com/yelp-dataset/yelp-dataset](https://www.kaggle.com/yelp-dataset/yelp-dataset)
2. Extract yelp_academic_dataset_review.json file the put in under ./WritingStyleTransfer/.data
3. Run buildData.py script
```sh
python buildData.py \
    --positive_review_stars_limit 5 \
    --negative_review_stars_limit 1 \
    --num_of_sentences_limit 10 \
    --min_num_of_words_limit 2 \
    --max_num_of_words_limit 15 \
    --test_size 0.2 \
    --validation_size 0.2 \
    --dataset_path ./.data/yelp_academic_dataset_review.json
```
Where
```
--positive_review_stars_limit - limit for positive reviews
--negative_review_stars_limit - limit for negative reviews
--num_of_sentences_limit - limit for the number of sentences in the review
--min_num_of_words_limit - limit for the minimum number of word in the review
--max_num_of_words_limit - limit for the maximum number of word in the review
--test_size - test set size
--validation_size - validation set size
--dataset_path - path of the dataset
```


### Train The Network

1. Run trainNetwork.py script
```sh
python trainNetwork.py \
    --batch_size 16 \
    --start_epoch 0 \
    --n_epochs 50 \
    --g_n_epochs 10 \
    --decay_epoch 25 \
    --lr 0.0002 \
    --lambda_cyc 8.0 \
    --lambda_id 5.0 \
    --lambda_adv 1.0 \
    --should_pretrain_generators True \
    --should_load_pretrain_generators False \
    --should_load_pretrain_discriminators False
```
Where
```
--batch_size - size of the batches
--start_epoch - epoch to start training from
--n_epochs - number of epochs for cycle GAN training
--g_n_epochs - number of epochs for generator training
--decay_epoch - epoch from which to start lr decay
--lr - adam: learning rate
--lambda_cyc - cycle loss weight
--lambda_id - identity loss weight
--lambda_adv - generator adversarial loss weight
--should_pretrain_generators - should pre train the generators
--should_load_pretrain_generators - should load the pre train generators
--should_load_pretrain_discriminators - should load the pre train discriminators
```

### See The Results
To see how well the network preform, you can open the DisplayNetwork.ipynb notebook in jupyter or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eliorav/WritingStyleTransfer/blob/master/DisplayNetwork.ipynb)

<!-- Hacks and Observations -->
## Hacks and Observations
* Construct different mini-batches for real and fake, i.e. each mini-batch needs to contain only all real images or all generated images.
* Use the ADAM Optimizer.
* Pre-train the generators.
* Use Dropouts in G in both train and test phase.
    * Provide noise in the form of dropout (50%).

<!-- References -->
## References
Here are some references we looked at while making these project.
### Papers
* [Generative Adversarial Nets](http://papers.nips.cc/paper/5423-generative-adversarial-nets)
* [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf)
* [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf)
* [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf)

### Other Resources
* [How to Train a GAN? Tips and tricks to make GANs work](https://github.com/soumith/ganhacks)

<!-- CONTACT -->
## Contact

* Elior Avraham - elior.av@gmail.com
* Natali Boniel - nataliboniel@gmail.com

Project Link: [https://github.com/eliorav/writing-style-transfer](https://github.com/eliorav/WritingStyleTransfer)
