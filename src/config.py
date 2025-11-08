
MODEL_NAME = 'bert-base-uncased'
NUM_LABELS = 2  

IMDB_DATASET_NAME = 'imdb'
BASELINE_MODEL_PATH = './models/baseline_model'


JIGSAW_COMPETITION_NAME = 'jigsaw-unintended-bias-in-toxicity-classification'
TOXICITY_THRESHOLD = 0.5
BIASED_MODEL_PATH = './models/biased_model'

EEC_DATASET_NAME = 'peixian/equity_evaluation_corpus'


TRAIN_EPOCHS = 5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 1e-3
DATA_DIR = './data'
RESULTS_DIR = './results'
JIGSAW_TRAIN_CSV = f'{DATA_DIR}/train.csv'
