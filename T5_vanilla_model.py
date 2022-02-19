from config_parser import config_args
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from nltk.tokenize import sent_tokenize



import torch
import os
from os.path import join
import pandas as pd
import numpy as np
import datasets
from TRBLL_dataset import TRBLLDataset
from datasets import load_metric


def get_actual_predictions(predictions, tokenized_dataset, tokenizer):
    """
    Get the actual predictions.
    The current method is to take the maximal value of the sub-words of a word as the prediction of the word
    :param predictions: list of predictions on sub-words
    :param tokenized_dataset: list of tokenized dataset
    :param tokenizer: tokenizer used for tokenization
    :return: predictions on original words
    """
    actual_predictions = []
    for i, (prediction_array, input_ids) in enumerate(zip(predictions.predictions, tokenized_dataset['input_ids'])):
        current_predictions = []
        prediction_array = prediction_array.argmax(-1)
        words = tokenizer.convert_ids_to_tokens(input_ids)
        for word_index, (word, prediction) in enumerate(zip(words, prediction_array)):
            if word != "[SEP]" and word != "[CLS]" and word.find("#") == -1:
                while word_index+1 < len(input_ids) and words[word_index+1].find("#") != -1:
                    prediction = max(prediction, prediction_array[word_index+1])
                    word_index += 1
                current_predictions.append(prediction)
        actual_predictions.append(current_predictions)
    return actual_predictions


def compute_metrics(eval_pred, tokenizer):
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


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # ROUGE expects a newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels


def preprocess_function(samples):
    max_input_length = 512
    max_target_length = 30
    model_name = "t5-small"  # config_args["model_vanilla_args"]["model_name"]
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # fix list of lists (mor)
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
    model_inputs["decoder_input_ids"] = labels["input_ids"]  # my addition
    return model_inputs


def run_model():
    model_name = "t5-small"  # config_args["model_vanilla_args"]["model_name"]
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5Model.from_pretrained(model_name)
    # model = T5ForConditionalGeneration.from_pretrained(model_name)

    samples_dataset = datasets.load_dataset('TRBLL_dataset.py')

    tokenized_datasets = samples_dataset.map(preprocess_function, batched=True)
    # args
    batch_size = 8
    num_train_epochs = 8
    # Show the training loss with every epoch
    logging_steps = len(tokenized_datasets["train"]) // batch_size

    args = Seq2SeqTrainingArguments(
        output_dir=f"{model_name}-finetuned-vanilla1",
        evaluation_strategy="epoch",
        learning_rate=5.6e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        logging_steps=logging_steps,
        push_to_hub=False,
    )


    # collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # remove the columns with strings because the collator won’t know how to pad these elements
    tokenized_datasets = tokenized_datasets.remove_columns(
        samples_dataset["train"].column_names
    )
    # the collator expects a list of dicts
    features = [tokenized_datasets["train"][i] for i in range(2)]
    data_collator(features)

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

    # trainer.train()
    # trainer.evaluate()

    #

    predictions = trainer.predict(tokenized_datasets["test"])
    predictions = get_actual_predictions(predictions, tokenized_datasets['test'], tokenizer)

if __name__ == '__main__':
    run_model()