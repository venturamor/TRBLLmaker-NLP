from transformers import GPT2LMHeadModel, GPT2Tokenizer,  GPTNeoForCausalLM
import datasets
import torch
import docx
import datetime
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model, T5TokenizerFast, GPT2Tokenizer, \
    GPT2LMHeadModel, Trainer, GPTNeoForCausalLM, TrainingArguments, TFGPT2LMHeadModel
from transformers import DataCollatorForSeq2Seq
from box import Box
import yaml
from config_parser import *
from tqdm import tqdm
from prompts import *
import pandas as pd
import numpy as np

folder = 'after_training'
# Load pickl as a dataframe
path = '/home/tok/TRBLLmaker/results/{}/'.format(folder)
pickle_name = 'predictions_after_training_2022-03-10-16-13-11.pkl'
df = pd.read_pickle(path + pickle_name)
#DF columns:
#'example_index', 'input_prompt', 'predicted_text', 'decode_method', 'temperature', 'model', 'prompt_type', 'meaning'


# create a doc file to write the generated prompts
doc = docx.Document()
doc.add_heading('Predicted annotations compare {}', 0)

# compare the same prompt and decode method with different models
# print the input prompt and the predicted text for each model
# input_prompt = df.loc[0, 'input_prompt']
# df_input_prompt = df[df['input_prompt'] == input_prompt]
# for index, row in df_input_prompt.iterrows():
#     para = doc.add_paragraph("model: {}\n".format(row['model']))
#     para.add_run("decode method: {}\n".format(row['decode_method']))
#     para.add_run("input prompt: {}\n".format(row['input_prompt']))
#     para.add_run("gt: {}\n".format(row['meaning']))
#     para.add_run("predicted text: {}\n".format(row['predicted_text'])).font.bold = True
#
# input_prompt = df.loc[5, 'input_prompt']
# df_input_prompt = df[df['input_prompt'] == input_prompt]
# for index, row in df_input_prompt.iterrows():
#     para = doc.add_paragraph("model: {}\n".format(row['model']))
#     para.add_run("decode method: {}\n".format(row['decode_method']))
#     para.add_run("input prompt: {}\n".format(row['input_prompt']))
#     para.add_run("gt: {}\n".format(row['meaning']))
#     para.add_run("predicted text: {}\n".format(row['predicted_text'])).font.bold = True

# Print all the prompts and the predicted text for each model
for index, row in df.iterrows():
    para = doc.add_paragraph("model: {}\n".format(row['model']), style='List Number')
    para.add_run("decode method: {}\n".format(row['decode_method']))
    para.add_run("input prompt: {}\n".format(row['input_prompt']))
    para.add_run("gt: {}\n".format(row['meaning']))
    para.add_run("predicted text: {}\n".format(row['predicted_text'])).font.bold = True

doc.save('/home/tok/TRBLLmaker/results/{}/{}_{}.docx'.format(folder, pickle_name.split('.')[0],
                                                             datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))

# compare the same prompt and model with different decode methods

# compare the same model and decode method with different prompts
