import argparse
import gc
import numpy as np
import torch
import wandb

from datasets import load_dataset, load_metric
from transformers import (
    BertTokenizer,
    BertModel,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

gc.collect()
torch.cuda.empty_cache()


def initialize_model(model_checkpoint, num_labels):
    """ Initializes the model and tokenizer for classification task

    @params:  model_checkpoint is the name of the model 
              num_labels is the number of classes
    """
    tokenizer = BertTokenizer.from_pretrained(model_checkpoint)
    model = BertForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    return tokenizer, model


def compute_metrics(eval_pred, task='xnli', language='en'):
    """ Returns the evaluation metric based on
    predictions and reference labels

    @params:  eval_pred is tuple of predictions and true labels
              task is huggingface metric utilized by model
              language is lang code for which model is evaluated
    """
    metric = load_metric(task, language)
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


def preprocess_function(examples, sentence1_key, sentence2_key, tokenizer):
    """ Processes the batch of dataset using the tokenizer

    @params:  examples is the batch of dataset
              sentence1_key is the first key in the dataset for input
              sentence2_key if notnull, is the first key in the dataset for input
              tokenizer is the tokenizer used
    """
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, default='en')
    args = parser.parse_args()

    print(f"Got language argument as {args.language}")
    NUM_LABELS = 3  # if classification task, how many classes
    BATCH_SIZE = 16  # batch_size for each training
    MODEL_CHECKPOINT = 'bert-base-multilingual-cased'  # model type
    # MODEL_CHECKPOINT = 'mbert-xnli-en-finetuned'
    DATASET = "xnli"  # dataset/corpus for which model needs to be finetuned
    NUM_EPOCHS = 5

    wandb.init(project=f"{args.language}-xnli-baseline")
    tokenizer, model = initialize_model(MODEL_CHECKPOINT, NUM_LABELS)
    dataset = load_dataset(DATASET, args.language)
    sentence1_key, sentence2_key = ("premise", "hypothesis")
    encoded_dataset = dataset.map(lambda p: preprocess_function(p, sentence1_key, sentence2_key, tokenizer),
                                  batched=True)

    huggingface_args = TrainingArguments(
        output_dir=f"baseline_langs/{args.language}",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="wandb"
    )

    validation_key = "test"

    trainer = Trainer(
        model,
        huggingface_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, DATASET, args.language)
    )
    trainer.train()
    print(trainer.evaluate())
    wandb.finish()

if __name__ == '__main__':
    main()
