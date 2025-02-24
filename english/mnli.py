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


def mnli_main(model_name_or_path, logger, output_dir, tokenizer_name_or_path=None):
    logger.info("Loading dataset...")
    raw_datasets = load_dataset("SetFit/mnli")
    raw_datasets.pop('test')
    train_test_split = raw_datasets['train'].train_test_split(test_size = 0.01,seed = 42)    
    raw_datasets['train'] = train_test_split['train']#.select(list(range(1, 8000)))
    raw_datasets['validation'] = raw_datasets['validation']#.select(list(range(1, 700)))
    raw_datasets['test'] = train_test_split['test']


    label_list = get_label_list(raw_datasets, split="train")
    label_list.sort()
    num_labels = len(label_list)

    if tokenizer_name_or_path is None:
        logger.info("Loading tokenizer from %s", model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    else:
        logger.info("Loading tokenizer from %s", tokenizer_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    label_to_id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    id_to_label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}


    logger.info("Loading model from %s", model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path,
                                                               num_labels=num_labels,
                                                               id2label=id_to_label,
                                                               label2id=label_to_id)
    
    def preprocess_function(examples):
       
        result = tokenizer(examples['text1'], examples['text2'], truncation=True, max_length=512)

        return result

    logger.info("Tokenizing dataset...")
    tokenized_dataset = raw_datasets.map(preprocess_function, batched=True, num_proc=8)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8) 

    training_args = TrainingArguments(
        output_dir=f"output_dir/{'mnli'}/",
        save_total_limit=1,
        learning_rate=5e-5,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        num_train_epochs=2,
        eval_steps=250,
        save_steps=250,
        logging_steps=50,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to='none',
        overwrite_output_dir=True,
        fp16=True,
        dataloader_num_workers=8
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

    logger.info("\n\n=================MNLI Dataset Results: \n")
    logger.info("Test results: %s", test_results)
