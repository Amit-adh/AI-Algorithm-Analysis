import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import (
    BertForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline
)
from tqdm.auto import tqdm
from . import config
from .data_loader import get_eec_dataset

def load_trained_model(model_type='baseline'):

    if model_type == 'baseline':
        path = config.BASELINE_MODEL_PATH
    elif model_type == 'biased':
        path = config.BIASED_MODEL_PATH
    else:
        raise ValueError("model_type must be 'baseline' or 'biased'")

    print(f"Loading trained model '{model_type}' from {path}")
    try:
        model = BertForSequenceClassification.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
    except OSError:
        print(f"Error: Model not found at {path}")
        print("Please make sure you have trained the model first.")
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, tokenizer

def get_logits(text, model, tokenizer):
    """ Gets the raw logits for a single piece of text. """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.logits[0].cpu().numpy()

def calculate_asd(model, tokenizer):

    print("Calculating Average Sentiment Difference (ASD)...")
    eec = get_eec_dataset()


    gender_data = eec['validation'].filter(lambda x: x['category'] == 'Gender')

    logit_diffs = []

    for item in tqdm(gender_data):
        logit1 = get_logits(item['sentence1'], model, tokenizer)
        logit2 = get_logits(item['sentence2'], model, tokenizer)


        diff = logit1[1] - logit2[1]
        logit_diffs.append(diff)

    asd = np.mean(np.abs(logit_diffs))
    print(f"Average Sentiment (Logit) Difference: {asd:.4f}")
    return asd

def plot_sentiment_shift(model, tokenizer):

    print("Plotting Sentiment Distribution Shift...")
    eec = get_eec_dataset()
    gender_data = eec['validation'].filter(lambda x: x['category'] == 'Gender')

    sample_data = gender_data.shuffle(seed=42).select(range(5))

    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=model.device, return_all_scores=True)

    for item in sample_data:
        sentences = [item['sentence1'], item['sentence2']]
        results = pipe(sentences)


        probs = [res[1]['score'] for res in results]
        labels = [item['stereotype'], item['anti-stereotype']]

        plt.figure(figsize=(6, 4))
        plt.bar(labels, probs, color=['blue', 'orange'])
        plt.title(f"Probability of 'Label 1' (Positive/Toxic)\nTemplate: {item['template']}")
        plt.ylabel("P(Label 1)")
        plt.ylim(0, 1)
        plt.show()
