import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split

df = pd.read_csv("/Users/stavyagaonkar/Downloads/AI_Human.csv")
df = df[['text', 'generated']].dropna()
df['generated'] = df['generated'].astype(int)

pd.set_option('display.max_colwidth', None)
print(df.iloc[40000])
