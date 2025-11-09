from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
import pandas as pd
from . import config
from .data_loader import get_jigsaw_dataframe



class JigsawDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }



def get_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_LABELS
    )
    return model, tokenizer


def train_biased_model():
    print("Starting BIASED model training process...")

    model, tokenizer = get_model_and_tokenizer()

    
    df = get_jigsaw_dataframe()
    if df is None:
        print(" Failed to load Jigsaw dataset. Please check Kaggle access.")
        return None, None

    
    df = df.sample(n=min(5000, len(df)), random_state=42).reset_index(drop=True)
    print(f"Loaded {len(df)} Jigsaw samples for training/testing.")

    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["comment_text"].values,
        df["labels"].values,
        test_size=0.2,
        random_state=42
    )

    
    train_dataset = JigsawDataset(train_texts, train_labels, tokenizer, max_len=config.MAX_SEQ_LEN)
    val_dataset = JigsawDataset(val_texts, val_labels, tokenizer, max_len=config.MAX_SEQ_LEN)

    
    training_args = TrainingArguments(
        output_dir=f"{config.RESULTS_DIR}/biased_checkpoints",
        num_train_epochs=config.TRAIN_EPOCHS,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        logging_dir=f"{config.RESULTS_DIR}/biased_logs",
        logging_steps=100,
        save_strategy="epoch",
        eval_strategy="epoch",  # Compatibility with your transformers version
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

    print("Training biased model...")
    trainer.train()

    # --- Save model ---
    print(f"Saving biased model to {config.BIASED_MODEL_PATH}")
    trainer.save_model(config.BIASED_MODEL_PATH)
    tokenizer.save_pretrained(config.BIASED_MODEL_PATH)
    print(" Biased model training complete.")

    return model, tokenizer
