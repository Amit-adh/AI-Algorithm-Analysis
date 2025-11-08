from transformers import (
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoTokenizer
)
from . import config
from .data_loader import get_jigsaw_dataframe, get_imdb_dataset, ToxicityDataset

def get_model_and_tokenizer():
   
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_LABELS
    )
    return model, tokenizer
def train_biased_model():
   
    print("Starting BIASED model training process...")
    model, tokenizer = get_model_and_tokenizer()

   
    df = get_jigsaw_dataframe()
    if df is None:
        print(" Failed to load Jigsaw data. Exiting training.")
        return None, None

   
    from sklearn.model_selection import train_test_split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['comment_text'].values, df['labels'].values, test_size=0.1, random_state=42
    )

    
    train_dataset = ToxicityDataset(train_texts, train_labels, tokenizer)
    val_dataset = ToxicityDataset(val_texts, val_labels, tokenizer)

    
    training_args = TrainingArguments(
        output_dir=f"{config.RESULTS_DIR}/biased_checkpoints",
        num_train_epochs=config.TRAIN_EPOCHS,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=f"{config.RESULTS_DIR}/biased_logs",
        logging_steps=100,
        report_to="none"
    )


    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("Training biased model...")
    trainer.train()


    print(f"Saving biased model to {config.BIASED_MODEL_PATH}")
    trainer.save_model(config.BIASED_MODEL_PATH)
    tokenizer.save_pretrained(config.BIASED_MODEL_PATH)
    print(" Biased model training complete.")
    return model, tokenizer


def train_baseline_model():
    
    print("Starting BASELINE model training process...")
    model, tokenizer = get_model_and_tokenizer()

 
    train_dataset, val_dataset = get_imdb_dataset()

  
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

   
    training_args = TrainingArguments(
    output_dir=f"{config.RESULTS_DIR}/baseline_checkpoints",
    num_train_epochs=config.TRAIN_EPOCHS,
    per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
    learning_rate=config.LEARNING_RATE,
    save_strategy="epoch",
    eval_strategy="epoch",             
    load_best_model_at_end=True,        
    logging_dir=f"{config.RESULTS_DIR}/baseline_logs",
    logging_steps=100,
    report_to="none"
)



    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("Training baseline model...")
    trainer.train()


    print(f"Saving baseline model to {config.BASELINE_MODEL_PATH}")
    trainer.save_model(config.BASELINE_MODEL_PATH)
    tokenizer.save_pretrained(config.BASELINE_MODEL_PATH)
    print(" Baseline training complete.")
    return model, tokenizer
