import numpy as np

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments, AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification

dataset = load_dataset("emotion", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding=False, truncation=True, max_length=512)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions), "f1": f1_score(labels, predictions, average="weighted")}


tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataset = tokenized_datasets["train"].shuffle(seed=42)
eval_dataset = tokenized_datasets["validation"].shuffle(seed=42)


model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=6)


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
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
