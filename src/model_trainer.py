from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from datasets import DatasetDict
from . import config
from .data_loader import get_imdb_dataset


def get_model_and_tokenizer():
    """Loads the pre-trained model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_LABELS
    )
    return model, tokenizer


def train_baseline_model():
    print("Starting BASELINE model training process...")

    model, tokenizer = get_model_and_tokenizer()

    
    dataset = get_imdb_dataset()

    
    if isinstance(dataset, tuple):
        print("Detected tuple dataset format — converting to DatasetDict...")
        dataset = DatasetDict({"train": dataset[0], "test": dataset[1]})
    elif not isinstance(dataset, DatasetDict):
        raise TypeError("Unsupported dataset format returned from get_imdb_dataset()")

   
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=config.MAX_SEQ_LEN
        )

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

   
    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(
        range(min(config.TRAIN_SUBSET_SIZE, len(tokenized_datasets["train"])))
    )
    val_dataset = tokenized_datasets["test"].shuffle(seed=42).select(
        range(min(config.VAL_SUBSET_SIZE, len(tokenized_datasets["test"])))
    )

 
    training_args = TrainingArguments(
        output_dir=f"{config.RESULTS_DIR}/baseline_checkpoints",
        num_train_epochs=config.TRAIN_EPOCHS,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        logging_dir=f"{config.RESULTS_DIR}/baseline_logs",
        logging_steps=100,
        save_strategy="epoch",
        eval_strategy="epoch",  
        load_best_model_at_end=True,
        fp16=config.USE_FP16,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        report_to="none"
    )

    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    print("Training baseline model...")
    trainer.train()

    
    print(f"Saving baseline model to {config.BASELINE_MODEL_PATH}")
    trainer.save_model(config.BASELINE_MODEL_PATH)
    tokenizer.save_pretrained(config.BASELINE_MODEL_PATH)
    print(" Baseline training complete.")

    return model, tokenizer
