# based on https://towardsdatascience.com/how-to-fine-tune-gpt-2-for-text-generation-ae2ea53bc272

import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
import os
import datetime
import torch.nn.functional as F
import csv


def load_data_and_split():
    # this dataset will be taken only from train!
    # json_path = r'C:\לימודים אקדמיים\תואר שני\עיבוד שפה טבעית\Final Project - TRBLLmaker\TRBLLmaker-NLP\data\samples\train.json'
    json_path = r'/home/student/mor_nlp/data/samples/validation_mini.json'
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


# create the dataset
class TRBLLDataset(Dataset):
    def __init__(self, txt_list, label_list, tokenizer, prompts, val_flag=False):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []

        max_length = 128

        for txt, label in tqdm(zip(txt_list, label_list)):
            # prepare text
            if val_flag:
                prep_txt = '{} {}\n{} '.format(prompts['text_prompt'], txt,
                                               prompts['label_prompt'])
            else:
                prep_txt = '{} {}\n{} {}'.format(prompts['text_prompt'], txt,
                                                 prompts['label_prompt'], label)
            # tokenize text
            tokenized_txt = tokenizer(prep_txt, truncation=True,
                                      max_length=max_length)  #, padding='max_length'
            # mor addition
            tokenized_label = tokenizer(label, truncation=True,
                                        max_length=max_length)  #, padding='max_length'

            # append
            self.input_ids.append(torch.tensor(tokenized_txt['input_ids']))
            self.attn_masks.append(torch.tensor(tokenized_txt.data['attention_mask']))
            # self.labels.append(label)
            self.labels.append(tokenized_label)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx] #, self.attn_masks[idx], self.labels[idx]


# Accumulated batch size (since GPT2 is so big)

def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, :]], dim=1)
        # packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None


def train(
    dataset, model, tokenizer,
    batch_size=16, epochs=2, lr=2e-5,
    max_seq_len=128, warmup_steps=50,
    gpt2_type="gpt2", output_dir=".",
    test_mode=False, save_model_on_epoch=False,
):

    output_prefix = gpt2_type + "_lr-" + str(lr) + "_bs-" + str(batch_size) + "_date-" + \
                      datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    acc_steps = 100
    device = torch.device("cuda")
    model = model.cuda()
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    loss = 0
    accumulating_batch_count = 0
    input_tensor = None

    for epoch in range(epochs):

        print(f"Training epoch {epoch}")
        print(loss)
        for idx, entry in tqdm(enumerate(train_dataloader)):
            (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, 768)

            if carry_on and idx != len(train_dataloader) - 1:
                continue

            # input_tensor = entry.to(device)

            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor, labels=input_tensor)
            loss = outputs[0]
            loss.backward()

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None
        if save_model_on_epoch:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            )
    return model


def generate(
        model,
        tokenizer,
        prompt,
        entry_count=10,
        entry_length=30,  # maximum number of words
        top_p=0.8,
        temperature=1.,
):
    model.eval()
    generated_num = 0
    generated_list = []

    filter_value = -float("Inf")

    with torch.no_grad():

        for entry_idx in trange(entry_count):

            entry_finished = False
            generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

            for i in range(entry_length):
                outputs = model(generated, labels=generated)
                loss, logits = outputs[:2]
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)

                if next_token in tokenizer.encode("<|endoftext|>"):
                    entry_finished = True

                if entry_finished:
                    generated_num = generated_num + 1

                    output_list = list(generated.squeeze().numpy())
                    output_text = tokenizer.decode(output_list)
                    generated_list.append(output_text)
                    break

            if not entry_finished:
                output_list = list(generated.squeeze().numpy())
                output_text = f"{tokenizer.decode(output_list)}<|endoftext|>"
                generated_list.append(output_text)

    return generated_list


# Function to generate multiple sentences. Test data should be a dataframe
def text_generation(model, X_test):
    # fit test_dataset to this script
    num_of_examples = 10
    generated_lyrics = []
    for txt in X_test[:num_of_examples]:
        prep_txt = '{} {}\n{} '.format(prompts['text_prompt'], txt,
                                         prompts['label_prompt'])
        x = generate(model.to('cpu'), tokenizer, prep_txt, entry_count=1)
        generated_lyrics.append(x)

    return generated_lyrics




if __name__ == '__main__':
    # Get the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    prompts = {'bos_token': '<startoftext>',
               'eos_token': '<endoftext>',
               'text_prompt': 'lyrics:',
               'label_prompt': 'meaning:'}

    # # load tokenizer
    # bos_token = '<startoftext>'
    # eos_token = '<endoftext>'
    pad_token = '<pad>'
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token=bos_token,
    #                                           eos_token=eos_token, pad_token=pad_token)
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token=pad_token)

    # load data
    X_train, X_test, X_val, y_train, y_test, y_val = load_data_and_split()
    # create dataset
    test_dataset = (X_test, y_test)
    # # generate before
    generated_lyrics_before = text_generation(model, X_test)

    dataset = TRBLLDataset(X_train, y_train, tokenizer, prompts)

    model = train(dataset, model, tokenizer)

    # Run the functions to generate the lyrics
    generated_lyrics = text_generation(model, X_test)

    print('done')