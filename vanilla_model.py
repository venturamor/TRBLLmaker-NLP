# ----------------------------- self explainations ----------------------------
# needs Model, Tokenizer and Config:

# class transformers.T5Model - outputting raw hidden-states without any specific head on top.
#   - uses class transformers.T5Config

# class transformers.T5ForConditionalGeneration - T5 Model with a language modeling head on top (of the decoder)
#    - for unsupervised ans supervised (with task prompt)
# https://huggingface.co/transformers/v3.0.2/model_doc/t5.html#t5forconditionalgeneration

# class transformers.T5EncoderModel - outputting encoder’s raw hidden-states without any specific head on top.

# Tokenizers:
# T5 Tokenizer - based on Unigram (subword tokenization algorithm by maximum prob of a given sentence)
# Fast T5 Tokenizer - based on sentencePiece (BPE [byte-pair encoding - chars most frequently] + Unigram), Google
# Inference only. no training

# ----------------------------- imports ----------------------------

from config_parser import config_args
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model
from transformers import DataCollatorForSeq2Seq
import torch
import os
from os.path import join
import pandas as pd
import datasets
from TRBLL_dataset import TRBLLDataset
# pip install sentencepiece

# ----------------------------- init ----------------------------

model_name = "t5-small"  # config_args["model_vanilla_args"]["model_name"]
tokenizer = T5Tokenizer.from_pretrained(model_name)
# model = T5Model.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# when generating, we will use the logits of right-most token to predict the next token
# so the padding should be on the left
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token  # to avoid an error

max_input_length = 512
max_target_length = 512
#######################################
# Dataset - my
samples_dataset = datasets.load_dataset('TRBLL_dataset.py')
data = samples_dataset['test']['data'][806]
labels = samples_dataset['test']['labels'][806]

#######################################
task_prefix = "generate_meaning: " #None  # "generate song lyrics meaning: "  # None

if task_prefix:
    inputs = tokenizer([task_prefix + sentence[0] for sentence in data], return_tensors="pt",
                       max_length=max_input_length, truncation=True, padding=True)
    labels = tokenizer([sentence[0] for sentence in labels], return_tensors="pt",
                       max_length=max_target_length, truncation=True,  padding=True)
else:
    inputs = tokenizer([sentence[0] for sentence in data], return_tensors="pt",
                       max_length=max_input_length, truncation=True, padding=True)
    labels = tokenizer([sentence[0] for sentence in labels], return_tensors="pt",
                       max_length=max_target_length, truncation=True,  padding=True)
#################
# https://towardsdatascience.com/asking-the-right-questions-training-a-t5-transformer-model-on-a-new-task-691ebba2d72c
model1 = T5Model.from_pretrained(model_name)
input_ids = inputs.input_ids
decoder_input_ids = labels.input_ids
preds = model1.predict([task_prefix + data[0]])

#################
# inputs: data {inputs_ids, attention mask}
output_sequences = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    do_sample=False,  # disable sampling to test if batching affects output, (greedy decoding otherwise)
)


df_labels = pd.DataFrame(columns=['lyrics', 'GT', 'generated'])
df_labels['GT'] = [sentence[0] for sentence in labels]
df_labels['lyrics'] = data
df_labels['generated'] = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
print('done')