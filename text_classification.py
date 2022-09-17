import datasets
import numpy as np
import pandas as pd
import random

from datasets import load_dataset, load_metric
from hfsharpness.nlpsharpness import BaseTrainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def compute_metrics(eval_pred):
    """ Returns the evaluation metric based on 
    predictions and reference labels

    @params:    eval_pred is tuple of predictions and true labels,
    """
    metric = load_metric("glue", "cola")
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)

def preprocess_function(examples, sentence1_key, sentence2_key, tokenizer):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

def main():
    GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
    task = "cola"
    model_checkpoint = "distilbert-base-uncased"
    batch_size = 16

    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_dataset("glue", actual_task)
    metric = load_metric('glue', actual_task)
    fake_preds = np.random.randint(0, 2, size=(64,))
    fake_labels = np.random.randint(0, 2, size=(64,))
    metric.compute(predictions=fake_preds, references=fake_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mnli-mm": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
    sentence1_key, sentence2_key = task_to_keys[task]

    encoded_dataset = dataset.map(lambda p: preprocess_function(p, sentence1_key, sentence2_key, tokenizer), batched=True)
    num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

    metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
    model_name = model_checkpoint.split("/")[-1]

    args = TrainingArguments(
        output_dir="./",
        num_train_epochs=1,              # total number of training epochs
        per_device_train_batch_size=128,  # batch size per device during training
        gradient_accumulation_steps=2,
        sam=True,                        # Use sharpness aware minimization
        sam_rho=0.01,                    # Step size for SAM
        fisher_penalty_weight=0.01,      # Use Fisher penalty with this weight
    )
    validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
    trainer = BaseTrainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    print(trainer.evaluate())

if __name__ == '__main__':
    main()
