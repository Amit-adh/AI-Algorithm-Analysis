-
MODEL_NAME = 'bert-base-uncased'
NUM_LABELS = 2

IMDB_DATASET_NAME = 'imdb'
BASELINE_MODEL_PATH = './models/baseline_model'


JIGSAW_COMPETITION_NAME = 'jigsaw-unintended-bias-in-toxicity-classification'
TOXICITY_THRESHOLD = 0.5
BIASED_MODEL_PATH = './models/biased_model'


EEC_DATASET_NAME = 'peixian/equity_evaluation_corpus'


TRAIN_EPOCHS = 20             
TRAIN_BATCH_SIZE = 8            
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
MAX_SEQ_LEN = 128              
TRAIN_SUBSET_SIZE =5000      
VAL_SUBSET_SIZE = 1000


USE_FP16 = True                 
GRADIENT_ACCUMULATION_STEPS = 2 


DATA_DIR = './data'
RESULTS_DIR = './results'
JIGSAW_TRAIN_CSV = f'{DATA_DIR}/train.csv'
