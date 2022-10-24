import datasets
import gc
import numpy as np
import pandas as pd
import random
import torch

from datasets import load_dataset, load_metric
from hfsharpness.nlpsharpness import BaseTrainer, TrainingArguments
from transformers import BertTokenizer, BertModel, BertForSequenceClassification

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

def compute_metrics(eval_pred):
    """ Returns the evaluation metric based on 
    predictions and reference labels

    @params:  eval_pred is tuple of predictions and true labels,
    """
    metric = load_metric("xnli", "en")
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
    NUM_LABELS = 3                                      # if classification task, how many classes
    BATCH_SIZE = 16                                     # batch_size for each training
    # MODEL_CHECKPOINT = 'bert-base-multilingual-cased'   # model type
    MODEL_CHECKPOINT = './checkpoint-30500'
    LANGUAGE = "en"                                     # language to be trained on
    DATASET = "xnli"                                    # dataset/corpus for which model needs to be finetuned
    FISHER_PENALTY_WEIGHT = 1.0
    USE_SAM = False
    NUM_EPOCHS = 5

    tokenizer, model = initialize_model(MODEL_CHECKPOINT, NUM_LABELS)
    dataset = load_dataset(DATASET, LANGUAGE)
    sentence1_key, sentence2_key = ("premise", "hypothesis")
    encoded_dataset = dataset.map(lambda p: preprocess_function(p, sentence1_key, sentence2_key, tokenizer), batched=True)
    args = TrainingArguments(
            output_dir="./",
            num_train_epochs=NUM_EPOCHS,                                # total number of training epochs
            per_device_train_batch_size=BATCH_SIZE,                     # batch size per device during training
            gradient_accumulation_steps=2,
            sam=USE_SAM,                                                # Use sharpness aware minimization
            sam_rho=0.01,                                               # Step size for SAM
            fisher_penalty_weight=FISHER_PENALTY_WEIGHT,                # Use Fisher penalty with this weight
        )
    validation_key = "test"

    trainer = BaseTrainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    print(f'Evaluation on test dataset using pre-trained model: {trainer.evaluate()}')

    
if __name__ == '__main__':
  main()
