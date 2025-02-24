import evaluate
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

set_seed(42)

def get_label_list(raw_dataset, split="train"):
    if isinstance(raw_dataset[split]["label"][0], list):
        label_list = [label for sample in raw_dataset[split]["label"] for label in sample]
        label_list = list(set(label_list))
    else:
        label_list = raw_dataset[split].unique("label")

    label_list = [str(label) for label in label_list]
    return label_list


accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy_result = accuracy.compute(predictions=predictions, references=labels)['accuracy']
    f1_result = f1.compute(predictions=predictions, references=labels, average='macro')['f1']
    return {"accuracy": accuracy_result, "f1": f1_result}


def arentail_main(model_name_or_path, logger, output_dir, tokenizer_name_or_path=None):
    logger.info("Loading dataset...")
    raw_datasets = load_dataset("arbml/ArEntail")
    train_val_split = raw_datasets['train'].train_test_split(test_size=0.1, seed=43)
    raw_datasets['train'] = train_val_split['train']
    raw_datasets['validation'] = train_val_split['test']

    label_list = get_label_list(raw_datasets, split="train")
    label_list.sort()
    num_labels = len(label_list)

    if tokenizer_name_or_path is None:
        logger.info("Loading tokenizer from %s", model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    else:
        logger.info("Loading tokenizer from %s", tokenizer_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    label_to_id = {'not entail': 0, 'entails': 1}
    id_to_label = {0: 'not entail', 1: 'entails'}


    logger.info("Loading model from %s", model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path,
                                                               num_labels=num_labels,
                                                               id2label=id_to_label,
                                                               label2id=label_to_id)
    
    def preprocess_function(examples):
       
        result = tokenizer(examples['premise'], examples['hypothesis'], truncation=True)

        return result

    logger.info("Tokenizing dataset...")
    tokenized_dataset = raw_datasets.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    training_args = TrainingArguments(
        output_dir=f"output_dir/{'arentail'}/",
        save_total_limit=2,
        learning_rate=5e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=7,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
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

    logger.info("\n\n=================ArEntail Dataset Results: \n")
    logger.info("Test results: %s", test_results)
