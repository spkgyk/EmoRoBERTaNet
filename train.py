import argparse
import numpy as np

from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    Trainer,
    AutoTokenizer,
    EvalPrediction,
    TrainingArguments,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
)

parser = argparse.ArgumentParser(description="Process model name.")
parser.add_argument("--model_name", type=str, help="The name of the model to use", default="distilbert/distilbert-base-uncased")
args = parser.parse_args()
model_name = args.model_name

dataset = load_dataset("emotion", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding=False, truncation=True, max_length=512)


def compute_metrics(eval_pred: EvalPrediction):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    logits = logits[0] if type(logits) == tuple else logits

    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": accuracy,
        "F1_score": f1,
        "precision": precision,
        "recall": recall,
    }


tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataset = tokenized_datasets["train"].shuffle(seed=42)
eval_dataset = tokenized_datasets["validation"].shuffle(seed=42)


training_args = TrainingArguments(
    output_dir=f"./results/{model_name}",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.001,
    logging_dir=f"./logs/{model_name}",
    logging_steps=25,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

trainer.train()

print("Training complete.")
