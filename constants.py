DATASET_NAME = 'YelpDatasets'

DATA_FOLDER = f'.data/{DATASET_NAME}'
MODELS_FOLDER = f'saved_models/{DATASET_NAME}'

POSITIVE_FILE_EXTENSION = '.pos'
NEGATIVE_FILE_EXTENSION = '.neg'

TRAIN_TYPE = 0
TEST_TYPE = 1
VALIDATION_TYPE = 2

DATASET_TYPES = {
    TRAIN_TYPE: 'train',
    TEST_TYPE: 'test',
    VALIDATION_TYPE: 'val',
}

MODEL_NAME_G_AB = 'G_AB'
MODEL_NAME_G_BA = 'G_BA'
MODEL_NAME_D_A = 'D_A'
MODEL_NAME_D_B = 'D_B'
MODEL_NAME_SENTIMENT_ANALYSIS = "SENTIMENT_ANALYSIS"

b1 = 0.5  # adam: decay of first order momentum of gradient
b2 = 0.999  # adam: decay of first order momentum of gradient
enc_emb_dim = 256
dec_emb_dim = 256
d_emb_dim = 256
g_hid_dim = 512
g_n_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5
g_clip = 1
