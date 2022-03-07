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

# Load pickl as a dataframe
pickle_name = 'filename.pkl'
gen_df = pd.read_pickle(pickle_name)

# Compare between different models and parameters

