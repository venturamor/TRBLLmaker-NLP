from config_parser import config_args
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model, T5TokenizerFast
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from nltk.tokenize import sent_tokenize
import wandb
import torch
import os
from os.path import join
import pandas as pd
import numpy as np
import datasets
from TRBLL_dataset import TRBLLDataset
from datasets import load_metric
import nltk
nltk.download('punkt')
from box import Box
import yaml


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # ROUGE expects a newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels


def preprocess_function(samples, tokenizer, max_input_length=512, max_target_length=512):
    # fix list of lists (Mor)
    samples["data"] = [sentence[0] for sentence in samples["data"]]
    samples["labels"] = [sentence[0] for sentence in samples["labels"]]

    model_inputs = tokenizer(
        samples["data"], max_length=max_input_length, truncation=True, padding=True
    )

    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            samples["labels"], max_length=max_target_length, truncation=True, padding=True
        )

    model_inputs["labels"] = labels["input_ids"]
    # model_inputs["decoder_input_ids"] = labels["input_ids"]  # my addition
    return model_inputs


def run_model():
    with open('config.yaml') as f:
        training_args = Box(yaml.load(f, Loader=yaml.FullLoader))

    learning_rate = training_args.train_args.learning_rate
    batch_size = training_args.train_args.batch_size

    model_name = "t5-small"  # config_args["model_vanilla_args"]["model_name"]
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    # model = T5Model.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    experiment_name = model_name + "_lr-" + str(learning_rate) + "_bs-" + str(batch_size)
    project_name = training_args.wandb_args.project_name
    entity = training_args.wandb_args.entity
    # initialize wandb to visualize training progress
    wandb.init(project=project_name, entity=entity, name=experiment_name)

    samples_dataset = datasets.load_dataset('TRBLL_dataset_mini.py')

    tokenized_datasets = samples_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=samples_dataset["train"].column_names,
        fn_kwargs={'tokenizer': tokenizer}
    )
    # args
    batch_size = 8
    num_train_epochs = 1
    # Show the training loss with every epoch
    logging_steps = len(tokenized_datasets["train"]) // batch_size
    args = Seq2SeqTrainingArguments(
        output_dir=f"{model_name}-finetuned-vanilla1",
        evaluation_strategy="steps",
        eval_steps=training_args.train_args.eval_steps,
        logging_strategy="steps",
        logging_steps=training_args.train_args.eval_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=training_args.train_args.weight_decay,
        save_total_limit=training_args.train_args.save_total_limit,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        push_to_hub=False,
        report_to="wandb",
        run_name=experiment_name,
    )


    # collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    #  TODO check if relevant
    # the collator expects a list of dicts
    # features = [tokenized_datasets["train"][i] for i in range(2)]
    # data_collator(features)

    def compute_metrics(eval_pred):
        # eval metric
        rouge_score = load_metric("rouge")

        predictions, labels = eval_pred
        # Decode generated summaries into text
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # Decode reference summaries into text
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # ROUGE expects a newline after each sentence
        decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
        # Compute ROUGE scores
        result = rouge_score.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        # Extract the median scores
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        return {k: round(v, 4) for k, v in result.items()}

    #  with the Trainer API
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.evaluate()
    predictions = trainer.predict(tokenized_datasets["validation"])
    # Save predictions to file
    with open(f"{experiment_name}_predictions.txt", "w") as f:
        for pred in predictions:
            f.write(pred + "\n")
    print(predictions)
    # predictions = get_actual_predictions(predictions, tokenized_datasets['validation'], tokenizer)
    # predictions = [pred.strip() for pred in predictions]


if __name__ == '__main__':
    run_model()
