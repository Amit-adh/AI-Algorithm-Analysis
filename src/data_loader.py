import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset
import torch
import os
import kaggle
import zipfile
from . import config


def download_jigsaw_data():

    os.makedirs(config.DATA_DIR, exist_ok=True)

    if os.path.exists(config.JIGSAW_TRAIN_CSV):
        print("Jigsaw data (train.csv) already exists. Skipping download.")
        return

    print("Downloading Jigsaw data...")
    try:
        kaggle.api.competition_download_files(
            config.JIGSAW_COMPETITION_NAME,
            path=config.DATA_DIR,
            quiet=False
        )

        zip_path = f"{config.DATA_DIR}/{config.JIGSAW_COMPETITION_NAME}.zip"
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract('train.csv', path=config.DATA_DIR)
        os.remove(zip_path)
        print("Data downloaded and unzipped.")
    except Exception as e:
        print(f"Error downloading/unzipping Jigsaw data: {e}")


def get_jigsaw_dataframe():

    try:
        df = pd.read_csv(config.JIGSAW_TRAIN_CSV)
    except FileNotFoundError:
        print("Jigsaw data not found. Downloading...")
        download_jigsaw_data()
        df = pd.read_csv(config.JIGSAW_TRAIN_CSV)

    df['labels'] = (df['target'] >= config.TOXICITY_THRESHOLD).astype(int)
    df = df[['comment_text', 'labels', 'target']].dropna()
    return df

class ToxicityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


from datasets import load_dataset

def get_imdb_dataset():
    print("Loading IMDb dataset...")
    dataset = load_dataset("imdb")
    dataset = dataset.rename_column("label", "labels")
    return dataset



def get_eec_dataset():

    print("Loading EEC dataset...")
    return load_dataset(config.EEC_DATASET_NAME, trust_remote_code=True)
