from config_parser import config_args
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model, T5TokenizerFast, GPT2Tokenizer, \
    GPT2LMHeadModel, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from nltk.tokenize import sent_tokenize
import wandb
import numpy as np
import datasets
from TRBLL_dataset import TRBLLDataset
from datasets import load_metric
import nltk
nltk.download('punkt')
from box import Box
import yaml
import datetime
from datasets import Dataset

# try to clean memory
import torch
torch.cuda.empty_cache()


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # ROUGE expects a newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels


def generate_prompts(samples, prompt_type, validation=False):
    with open('config.yaml') as f:
        training_args = Box(yaml.load(f, Loader=yaml.FullLoader))
    if prompt_type == "constant":
        if not validation:
            data = [training_args.train_args.prompt.text + " " + sentence[0] + '. ' + label[0]
                    for (sentence, label) in zip(samples["data"], samples["labels"])]
        else:
            data = [training_args.train_args.prompt.text + " " + sentence[0] + '.'
                    for sentence in samples["data"]]
    elif prompt_type == "song_metadata":
        # Load the songs and annotations
        if not validation:
            data = ["Explain the next line from the song " + '"' + title[0] + '" by ' + artist[0] + ": "
                    + sentence[0] + '. ' + label[0] for (artist, title, sentence, label) in
                    zip(samples["artist"], samples["title"], samples["data"], samples["labels"])]
        else:
            data = ["Explain the next line from the song " + '"' + title[0] + '" by ' + artist[0] + ": "
                    + sentence[0] + '.' for (artist, title, sentence) in
                    zip(samples["artist"], samples["title"], samples["data"])]
    elif prompt_type == "question_context":
        if not validation:
            data = ["question: what is the meaning of " + artist[0] + " in the song " + '"' + title[0] + '"? ' +
                    "context: " + sentence[0] + '. answer:' + label[0]
                    for (artist, title, sentence, label) in
                    zip(samples["artist"], samples["title"], samples["data"], samples["labels"])]
        else:
            data = ["question: what is the meaning of " + artist[0] + " in the song " + '"' + title[0] + '"? ' +
                    "context: " + sentence[0] + '.' + "answer:"
                    for (artist, title, sentence) in zip(samples["artist"], samples["title"], samples["data"])]
    else:  # default: no prompt
        if not validation:
            data = samples["data"] + samples["labels"]
        else:
            data = samples["data"]
    return data


def preprocess_function_gpt2(samples, tokenizer, max_input_length=512, max_target_length=512):
    with open('config.yaml') as f:
        training_args = Box(yaml.load(f, Loader=yaml.FullLoader))
    if training_args.train_args.prompt.add_prompt:
        samples["data"] = generate_prompts(samples, prompt_type=training_args.train_args.prompt.prompt_type)
    else:
        # fix list of lists
        samples["data"] = [sentence[0] + '. ' + label[0] for (sentence, label) in zip(samples["data"], samples["label"])]

    samples["data"] = ['<|startoftext|>' + sentence + '<|endoftext|>' for sentence in samples["data"]]

    model_inputs = tokenizer(
        samples["data"], max_length=max_input_length, truncation=True, padding="longest"
    )
    model_inputs['attention_mask'] = model_inputs['attention_mask']
    model_inputs['labels'] = model_inputs['input_ids']

    return model_inputs


def preprocess_function_gpt2_validation(samples, tokenizer, max_input_length=512, max_target_length=512):
    with open('config.yaml') as f:
        training_args = Box(yaml.load(f, Loader=yaml.FullLoader))
    if training_args.train_args.prompt.add_prompt:
        samples["data"] = generate_prompts(samples, prompt_type=training_args.train_args.prompt.prompt_type, validation=True)
    else:
        # fix list of lists
        samples["data"] = [sentence[0] + '.' for (sentence, label) in samples["data"]]

    samples["data"] = ['<|startoftext|>' + sentence + '<|endoftext|>' for sentence in samples["data"]]

    model_inputs = tokenizer(
        samples["data"], max_length=max_input_length, truncation=True, padding="longest"
    )
    model_inputs['attention_mask'] = model_inputs['attention_mask']
    model_inputs['labels'] = model_inputs['input_ids']

    return model_inputs


def run_model():
    with open('config.yaml') as f:
        training_args = Box(yaml.load(f, Loader=yaml.FullLoader))

    learning_rate = training_args.train_args.learning_rate
    batch_size = training_args.train_args.batch_size
    num_train_epochs = training_args.train_args.num_train_epochs

    model_name = 'gpt2-medium'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name,  bos_token='<|startoftext|>', eos_token='<|endoftext|>',
                                              pad_token='<|pad|>')

    model = GPT2LMHeadModel.from_pretrained(model_name).cuda()
    model.config.update({"max_length": 512})
    model.resize_token_embeddings(len(tokenizer))

    experiment_name = model_name + "_lr-" + str(learning_rate) + "_bs-" + str(batch_size) +"_date-"+ \
                      datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    project_name = training_args.wandb_args.project_name
    entity = training_args.wandb_args.entity

    # initialize wandb to visualize training progress
    wandb.init(project=project_name, entity=entity, name=experiment_name)

    samples_dataset = datasets.load_dataset(training_args.train_args.dataset_name)

    tokenized_train = samples_dataset['train'].map(
        preprocess_function_gpt2,
        batched=True,
        remove_columns=samples_dataset["train"].column_names,
        fn_kwargs={'tokenizer': tokenizer}
    )

    tokenized_validation = samples_dataset['validation'].map(
        preprocess_function_gpt2_validation,
        batched=True,
        remove_columns=samples_dataset["train"].column_names,
        fn_kwargs={'tokenizer': tokenizer}
    )

    tokenized_test = samples_dataset['test'].map(
        preprocess_function_gpt2_validation,
        batched=True,
        remove_columns=samples_dataset["train"].column_names,
        fn_kwargs={'tokenizer': tokenizer}
    )
    # args

    # Show the training loss with every epoch
    # logging_steps = len(tokenized_datasets["train"]) // batch_size

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
        examples_to_print = 10
        # Print the first examples to the end of training_eval.txt
        with open("training_eval.txt", "a") as f:
            f.write("###################Eval#####################: " + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "\n")
            for i in range(examples_to_print):
                f.write("Prediction: {}\n".format(decoded_preds[i]))
                f.write("Label: {}\n".format(decoded_labels[i]))
                f.write("\n\n\n")
        for i in range(examples_to_print):
            print(f"Prediction: {decoded_preds[i]}")
            print(f"Label: {decoded_labels[i]}")
        # Compute ROUGE scores
        result = rouge_score.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        # Extract the median scores
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        return {k: round(v, 4) for k, v in result.items()}

    #  with the Trainer API
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    #  Eval the model before training
    for sample in samples_dataset["validation"]:
        print("Strating generation...")
        data = sample['data'][0]
        label = sample['labels'][0]
        title = sample['title'][0]
        artist = sample['artist'][0]
        txt = "question: what is the meaning of " + artist + " in the song " + '"' + title + '"? ' +\
              "context: " + data + '.' + "answer:"
        txt = '<|startoftext|>' + txt + '<|endoftext|>'
        generated = tokenizer(txt, return_tensors="pt").input_ids.cuda()
        sample_outputs = model.generate(generated, do_sample=True, top_k=50,
                                        max_length=300, top_p=0.95, temperature=2.0, num_return_sequences=5)
        print("Generation done.")
        # Decode generated summaries into text
        # Print index, text, and label
        for i, sample_output in enumerate(sample_outputs):
            print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
            print("{}: {}".format(i, label))
            break



    trainer.train()

    #  Eval the model after training
    for sample in samples_dataset["validation"]:
        print("Strating generation...")
        data = sample['data'][0]
        label = sample['labels'][0]
        title = sample['title'][0]
        artist = sample['artist'][0]
        txt = "question: what is the meaning of " + artist + " in the song " + '"' + title + '"? ' +\
              "context: " + data + '.' + "answer:"
        txt = '<|startoftext|>' + txt + '<|endoftext|>'
        generated = tokenizer(txt, return_tensors="pt").input_ids.cuda()
        sample_outputs = model.generate(generated, do_sample=True, top_k=50,
                                        max_length=300, top_p=0.95, temperature=0.8, num_return_sequences=5)
        print("Generation done.")
        # Decode generated summaries into text
        # Print index, text, and label
        for i, sample_output in enumerate(sample_outputs):
            print("{}: {} {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True), label))

    trainer.evaluate()
    predictions = trainer.predict(tokenized_validation)
    predictions_text = tokenizer.batch_decode(predictions[0], skip_special_tokens=True)

    # Save predictions to file
    with open(f"{experiment_name}_predictions" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".txt", "w") as f:
        f.write("Predictions: \n")
        for index, (song, annotation, prediction) in enumerate(zip(samples_dataset["validation"]["data"],
                                                                   samples_dataset["validation"]["labels"],
                                                                   predictions_text)):
            #  Print in seperated lines: index, song, annotation, prediction
            f.write(f"{index}\n")
            f.write("Song: " + str(song) + "\n")
            f.write("Annotation: " + str(annotation) + "\n")
            f.write("Prediction: " + str(prediction) + "\n")
            f.write("\n")
            f.write("\n")

    # Save the model
    trainer.save_model(training_args.train_args.results_checkpoints_dir + "/" + experiment_name)
    print("Saved model to {}".format(training_args.train_args.results_checkpoints_dir + "/" + experiment_name))


if __name__ == '__main__':
    run_model()
