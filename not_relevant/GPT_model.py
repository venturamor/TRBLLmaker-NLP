import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM, Trainer, Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments, TrainingArguments
from datasets import load_metric
from box import Box
import yaml
import datetime
import wandb
from tqdm import tqdm
import re
import numpy as np
from nltk.tokenize import sent_tokenize


class TRBLLDataset(Dataset):
    def __init__(self, txt_list, label_list, tokenizer, prompts, val_flag=False):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []

        max_length = 128

        for txt, label in tqdm(zip(txt_list, label_list)):
            # prepare text
            if val_flag:
                prep_txt = '{}{} {}\n{} '.format(prompts['bos_token'], prompts['text_prompt'], txt,
                                                 prompts['label_prompt'])
            else:
                prep_txt = '{}{} {}\n{} {}{}'.format(prompts['bos_token'], prompts['text_prompt'], txt,
                                                     prompts['label_prompt'], label, prompts['eos_token'])
            # tokenize text
            tokenized_txt = tokenizer(prep_txt, truncation=True,
                                      max_length=max_length, padding='max_length')
            # mor addition
            tokenized_label = tokenizer(label, truncation=True,
                                        max_length=max_length, padding='max_length')

            # append
            self.input_ids.append(torch.tensor(tokenized_txt['input_ids']))
            self.attn_masks.append(torch.tensor(tokenized_txt.data['attention_mask']))
            # self.labels.append(label)
            self.labels.append(tokenized_label)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.labels[idx]


def load_data_and_split():
    # this dataset will be taken only from train!
    # json_path = r'C:\לימודים אקדמיים\תואר שני\עיבוד שפה טבעית\Final Project - TRBLLmaker\TRBLLmaker-NLP\data\samples\train.json'
    json_path = r'/home/tok/TRBLLmaker/data/samples/validation_mini.json'
    # json_path = r'/home/student/mor_nlp/data/samples/train.json'
    df_samples_orig = pd.read_json(json_path)
    data_col = "text"
    label_col = "annotation"
    test_size = 0.2
    df_samples = df_samples_orig[[data_col, label_col]]
    # df_samples.drop(axis=0, index=df_samples.shape[0])  # so test size will be fit
    # df_samples.shape[0]

    X_train, X_test, y_train, y_test = train_test_split(df_samples[data_col].tolist(),
                                                        df_samples[label_col].tolist(),
                                                        shuffle=True, test_size=test_size,
                                                        random_state=1)  # stratify=df_samples[label_col]

    # validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      shuffle=True, test_size=test_size,
                                                      random_state=1)

    return X_train, X_test, X_val, y_train, y_test, y_val


def run_inference(tokenizer, model, test_dataset, prompts):
    # run model inference on all test data
    original_label = []
    predicted_label = []
    original_text = []
    predicted_text = []

    top_k = 50
    max_target_length = 128
    top_p = 0.9
    temprature = 0.9
    num_return_sequence = 1

    print_every = 10
    count = 0

    decode_method = 'sampling'
    decode_methods = ['greedy', 'beam search', 'sampling', 'top-k sampling', 'top-p sampling']
    # assert decode_method in decode_methods, 'Decode method should be one of the list'

    for txt, label in tqdm(zip(test_dataset[0], test_dataset[1])):
        count += 1
        # create prompt - on compliance with the one used during training
        prep_txt = '{}{} {}\n{} '.format(prompts['bos_token'], prompts['text_prompt'], txt,
                                         prompts['label_prompt'])
        tokenized_txt = tokenizer(f"{prep_txt}", return_tensors="pt").input_ids.cuda()

        # predict
        if decode_method == 'greedy':
            # greedy
            outputs = model.generate(tokenized_txt, max_length=max_target_length,
                                     temperature=temprature)  # num_return_sequence=num_return_sequence
        elif decode_method == 'beam search':
            # beam search with penalty on repeat
            outputs = model.generate(tokenized_txt, max_length=max_target_length,
                                     num_beams=3, early_stopping=True,
                                     no_repeat_ngram_size=2)
        elif decode_method == 'sampling':
            # sampling
            outputs = model.generate(tokenized_txt, do_sample=True, max_length=max_target_length,
                                     temperature=temprature)  # top_k=0,
        elif decode_method == 'top-k sampling':
            # top-k sampling
            outputs = model.generate(tokenized_txt, do_sample=True, top_k=50, max_length=max_target_length,
                                     )
        elif decode_method == 'top-p sampling':
            # top-p sampling
            outputs = model.generate(tokenized_txt, do_sample=True, top_k=0, top_p=0.92,
                                     max_length=max_target_length,
                                     )
        # additional paramters for better decoding - # repetition_penalty, min_length

        # decode
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # # extract the predicted sentiment
        # try:
        #     pred_meaning = re.findall('\n{} (.*)'.format(prompts['label_prompt']), predicted_text)[-1]
        # except:
        #     pred_meaning = 'None'

        # append results
        original_label.append(label)
        # predicted_label.append(pred_meaning)
        original_text.append(txt)
        predicted_text.append(pred_text)

        if count % print_every == 0:
            print('Original Lyrics:', txt, '\n',
                  'Predicted:', pred_text, '\n',
                  'Original Label:', label, '\n',
                  # 'Predicted Label:', pred_meaning, '\n'
                  )

    df_inference = pd.DataFrame({'original_text': original_text, 'predicted_meaning': predicted_label,
                                 'original_meaning': original_label, 'predicted_text': predicted_text})

    return df_inference


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
    examples_to_print = 5
    # Print the first examples to the end of training_eval.txt
    with open("training_eval.txt", "a") as f:
        f.write(
            "################### Eval #####################: " + datetime.datetime.now().strftime(
                "%Y%m%d-%H%M%S") + "\n")
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


if __name__ == '__main__':

    # define model parameters
    with open('config.yaml') as f:
        config_args = Box(yaml.load(f, Loader=yaml.FullLoader))

    model_name = "gpt2"

    learning_rate = config_args.train_args.learning_rate
    batch_size = config_args.train_args.batch_size
    num_train_epochs = config_args.train_args.num_train_epochs

    # params for run saving
    experiment_name = model_name + "_lr-" + str(learning_rate) + "_bs-" + str(batch_size) + "_date-" + \
                      datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    project_name = config_args.wandb_args.project_name
    entity = config_args.wandb_args.entity

    # initialize wandb to visualize training progress
    wandb.init(project=project_name, entity=entity, name=experiment_name,
               settings=wandb.Settings(start_method="fork"))

    # seed
    torch.manual_seed(42)

    # load tokenizer
    bos_token = '<startoftext>'
    eos_token = '<endoftext>'
    pad_token = '<pad>'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, bos_token=bos_token,
                                              eos_token=eos_token, pad_token=pad_token)

    # model
    if model_name == "gpt2" or model_name == 'gpt2-medium':
        model = GPT2LMHeadModel.from_pretrained(model_name).cuda()
    elif model_name == "EleutherAI/gpt-neo-1.3B":
        model = GPTNeoForCausalLM.from_pretrained(model_name).cuda()
    else:
        assert 'Error: This model is not gpt2 or gpt-neo'

    model.resize_token_embeddings(len(tokenizer))

    # prompts
    prompts = {'bos_token': '<startoftext>',
               'eos_token': '<endoftext>',
               'text_prompt': 'lyrics:',
               'label_prompt': 'meaning:'}

    # load data
    X_train, X_test, X_val, y_train, y_test, y_val = load_data_and_split()
    # create dataset
    train_dataset = TRBLLDataset(X_train, y_train, tokenizer, prompts)
    test_dataset = (X_test, y_test)
    # validation_dataset = (X_val, y_val)
    validation_dataset = TRBLLDataset(X_val, y_val, tokenizer, prompts, val_flag=True)

    # # inference before training
    df_inference_before = run_inference(tokenizer, model, test_dataset, prompts)

    # training args
    training_args = TrainingArguments(output_dir=f"{model_name}-finetuned-vanilla1",
                                      evaluation_strategy="steps",
                                      eval_steps=config_args.train_args.eval_steps,
                                      logging_strategy="steps",
                                      logging_steps=config_args.train_args.eval_steps,
                                      # load_best_model_at_end=True,
                                      learning_rate=learning_rate,
                                      per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      weight_decay=config_args.train_args.weight_decay,
                                      num_train_epochs=num_train_epochs,
                                      # warmup_steps=100,
                                      # predict_with_generate=True,
                                      save_total_limit=config_args.train_args.save_total_limit,
                                      report_to="wandb",
                                      run_name=experiment_name)
    # training_args = Seq2SeqTrainingArguments(output_dir=f"{model_name}-finetuned-vanilla1",
    #                                          evaluation_strategy="steps",
    #                                          eval_steps=config_args.train_args.eval_steps,
    #                                          logging_strategy="steps",
    #                                          logging_steps=config_args.train_args.eval_steps,
    #                                          # load_best_model_at_end=True,
    #                                          learning_rate=learning_rate,
    #                                          per_device_train_batch_size=batch_size,
    #                                          per_device_eval_batch_size=batch_size,
    #                                          weight_decay=config_args.train_args.weight_decay,
    #                                          num_train_epochs=num_train_epochs,
    #                                          # warmup_steps=100,
    #                                          predict_with_generate=True,
    #                                          save_total_limit=config_args.train_args.save_total_limit,
    #                                          report_to="wandb",
    #                                          run_name=experiment_name)

    # Trainer
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=validation_dataset,
                      data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                  'attention_mask': torch.stack([f[1] for f in data]),
                                                  'labels': torch.stack([f[0] for f in data])},
                      compute_metrics=compute_metrics)
    # trainer = Seq2SeqTrainer(model=model,
    #                          args=training_args,
    #                          train_dataset=train_dataset,
    #                          eval_dataset=validation_dataset,
    #                          data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
    #                                                      'attention_mask': torch.stack([f[1] for f in data]),
    #                                                      'labels': torch.stack([f[0] for f in data])},
    #                          compute_metrics=compute_metrics)

    # trainer.train()

    labels = validation_dataset.labels
    input_ids = validation_dataset.input_ids
    predictions = model.generate(input_ids, num_return_sequences=1)
    generated_list = tokenizer.batch_decode(predictions)
    compute_metrics(predictions, labels)

    trainer.evaluate()

    # test

    _ = model.eval()

    df_inference = run_inference(tokenizer, model, test_dataset, prompts)

    # print few examples from inference

    print('done')
