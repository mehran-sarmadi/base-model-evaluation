import evaluate
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from qa_utils.trainer_qa import QuestionAnsweringTrainer
from qa_utils.utils_qa import postprocess_qa_predictions

from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    TrainingArguments,
    set_seed,
)
import datetime


set_seed(42)

metric = evaluate.load("squad")

def compute_metrics(eval_pred):
    return metric.compute(predictions=eval_pred.predictions, references=eval_pred.label_ids)


def squad_main(model_name_or_path, logger, output_dir, tokenizer_name_or_path=None):
    logger.info("Loading dataset...")
    raw_datasets = load_dataset("rajpurkar/squad")
    train_test_split = raw_datasets['train'].train_test_split(test_size = 0.01, seed=42)    
    raw_datasets['train'] = train_test_split['train']
    raw_datasets['validation'] =  raw_datasets['validation']
    raw_datasets['test'] =  train_test_split['test']

    if tokenizer_name_or_path is None:
        logger.info("Loading tokenizer from %s", model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    else:
        logger.info("Loading tokenizer from %s", tokenizer_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    logger.info("Loading model from %s", model_name_or_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)
    

    column_names = raw_datasets["train"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    pad_on_right = tokenizer.padding_side == "right"


    def prepare_train_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=512,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            if tokenizer.cls_token_id in input_ids:
                cls_index = input_ids.index(tokenizer.cls_token_id)
            elif tokenizer.bos_token_id in input_ids:
                cls_index = input_ids.index(tokenizer.bos_token_id)
            else:
                cls_index = 0

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples
        # Validation preprocessing
    def prepare_validation_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=512,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    logger.info("Tokenizing dataset...")
    train_dataset = raw_datasets["train"].map(
                prepare_train_features,
                batched=True,
                num_proc=8,
                remove_columns=column_names,
                desc="Running tokenizer on train dataset",
            )

    eval_dataset = raw_datasets["validation"].map(
                prepare_validation_features,
                batched=True,
                num_proc=8,
                remove_columns=column_names,
                desc="Running tokenizer on validation dataset",
            )
    test_dataset = raw_datasets["test"].map(
                prepare_validation_features,
                batched=True,
                num_proc=8,
                remove_columns=column_names,
                desc="Running tokenizer on validation dataset",
            )
    


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8) 


    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            output_dir=training_args.output_dir,
            prefix=stage,
        )
        formatted_predictions = [{"id": str(k), "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": str(ex["id"]), "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    training_args = TrainingArguments(
        output_dir=f"results/models/{'squad'}/{str(datetime.datetime.now())}",
        save_total_limit=1,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=1,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to='none',
        overwrite_output_dir=True,
        fp16=True,
        metric_for_best_model='eval_f1',
    )

    logger.info("Initializing Trainer...")
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=raw_datasets['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Evaluating model on the test dataset...")
    test_results = trainer.evaluate(eval_dataset=test_dataset, eval_examples=raw_datasets['test'])

    logger.info("\n\n=================SQuAD Dataset Results: \n")
    logger.info("Test results: %s", test_results)