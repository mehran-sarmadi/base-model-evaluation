import evaluate
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
import datetime

from transformers import (
    AutoModelForMultipleChoice,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

import os
from typing import Dict, Optional, Union, List

import torch
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
from datasets import load_dataset, Features, Value, ClassLabel, Sequence
from datasets.formatting.formatting import LazyBatch
from dataclasses import dataclass
import evaluate
from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy


set_seed(42)


def compute_metrics(eval_pred):
    prediction, labels = eval_pred
    prediction = np.argmax(prediction, axis=1)
    return {
        "accuracy": np.mean(prediction == labels),
        "f1-weighted": f1_score(labels, prediction, average='weighted'),
        "precision": precision_score(labels, prediction, average='weighted'),
        "recall": recall_score(labels, prediction, average='weighted')
    }

@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)]
            for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def race_main(model_name_or_path, logger, output_dir, tokenizer_name_or_path=None):
    logger.info("Loading dataset...")
    raw_datasets = load_dataset("ehovy/race", 'middle')
    raw_datasets.shuffle()


    if tokenizer_name_or_path is None:
        logger.info("Loading tokenizer from %s", model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    else:
        logger.info("Loading tokenizer from %s", tokenizer_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)


    logger.info("Loading model from %s", model_name_or_path)
    model = AutoModelForMultipleChoice.from_pretrained(model_name_or_path)
    
    # Convert labels to IDs
    labels_to_id = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
    }

    def preprocess_function(examples: LazyBatch[str, List[str]]) -> Dict:
        # Repeat the context for each ending
        first_sentences = [
            [context] * 4
            for context in examples['article']
        ]

        # Extract question headers
        question_headers = examples['question']

        # Combine question headers with each possible ending
        second_sentences = [
            [f"{header} {examples['options'][i][idx]}" for idx in range(4)] 
            for i, header in enumerate(question_headers)
        ]

        # Flatten the sentences
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        # Tokenize the sentences
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=512
        )


        tokenized_labels = [labels_to_id[label] for label in examples['answer']]

        # Return the tokenized examples and labels
        tokenized_examples = {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        tokenized_examples['labels'] = tokenized_labels

        return tokenized_examples


    # logger.info("Tokenizing dataset...")
    tokenized_dataset = raw_datasets.map(
        preprocess_function, batched=True, num_proc=8
    )


    data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer, max_length=512)

    training_args = TrainingArguments(
        output_dir=f"results/models/{'Race'}/{str(datetime.datetime.now())}",
        save_total_limit=1,
        learning_rate=5e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1-weighted",
        report_to='none',
        overwrite_output_dir=True,
        fp16=True,
    )

    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Evaluating model on the test dataset...")
    test_results = trainer.evaluate(eval_dataset=tokenized_dataset['test'])

    logger.info("\n\n=================ParsiNLU Dataset Results: \n")
    logger.info("\n" * 2 + "*" * 30 + "\n" + f"Test results: {test_results}" + "\n" + "*" * 30 + "\n" * 2)
