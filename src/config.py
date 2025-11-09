
# --- Model Config ---
MODEL_NAME = 'bert-base-uncased'
NUM_LABELS = 2

# --- Dataset Config ---
IMDB_DATASET_NAME = 'imdb'
BASELINE_MODEL_PATH = './models/baseline_model'

# --- Jigsaw / Biased Model Config ---
JIGSAW_COMPETITION_NAME = 'jigsaw-unintended-bias-in-toxicity-classification'
TOXICITY_THRESHOLD = 0.5
BIASED_MODEL_PATH = './models/biased_model'

# --- Evaluation Dataset ---
EEC_DATASET_NAME = 'peixian/equity_evaluation_corpus'

# --- Training Config ---
# Optimized for speed
TRAIN_EPOCHS = 20              # still 10 epochs
TRAIN_BATCH_SIZE = 8            # smaller batch fits better in GPU memory
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
MAX_SEQ_LEN = 128               # shorter sentences = faster training

# --- Data Sampling ---
# Train on a subset of IMDb to reduce time (~20% of full dataset)
TRAIN_SUBSET_SIZE =5000       # full is 25k, 10k is fast & still valid
VAL_SUBSET_SIZE = 1000

# --- Training Optimizations ---
USE_FP16 = True                 # Mixed precision (speeds up on CUDA)
GRADIENT_ACCUMULATION_STEPS = 2 # Simulate larger batch without slowing down

# --- Directories ---
DATA_DIR = './data'
RESULTS_DIR = './results'
JIGSAW_TRAIN_CSV = f'{DATA_DIR}/train.csv'
