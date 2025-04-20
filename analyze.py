import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split

df = pd.read_csv("/Users/stavyagaonkar/Downloads/AI_Human.csv")
df = df[['text', 'generated']].dropna()
df['generated'] = df['generated'].astype(int)

train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

train_ds = Dataset.from_pandas(train_df)
test_ds  = Dataset.from_pandas(test_df)

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

train_ds = train_ds.map(tokenize, batched=True)
test_ds  = test_ds.map(tokenize,  batched=True)

train_ds = train_ds.rename_column("generated", "labels")
test_ds  = test_ds.rename_column("generated", "labels")

train_ds = train_ds.remove_columns(["text"])
test_ds  = test_ds.remove_columns(["text"])

train_ds.set_format("torch", ["input_ids", "attention_mask", "labels"])
test_ds .set_format("torch", ["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

training_args = TrainingArguments(
    output_dir="./model_output",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)

trainer.train()

model.save_pretrained("saved_llm_model")
tokenizer.save_pretrained("saved_llm_model")
