import kagglehub
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import evaluate
import torch
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu

df = pd.read_csv("data/final.csv")
df = df.rename(columns={'t': 'modern', 'og': 'shakespeare'})
print(df)