from transformers import GPT2LMHeadModel, GPT2Tokenizer,  GPTNeoForCausalLM
import datasets
import torch
import docx
import datetime
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model, T5TokenizerFast, GPT2Tokenizer, \
    GPT2LMHeadModel, Trainer, GPTNeoForCausalLM, TrainingArguments
from transformers import DataCollatorForSeq2Seq
from box import Box
import yaml
from config_parser import *
from tqdm import tqdm


def generate_prompts(lyrics, meaning, artist="artist", title="song", prompt_type=None):
    if prompt_type == "lyrics_meaning":
        data = "lyrics: {}.\n meaning:".format(lyrics)
    elif prompt_type == "song_metadata":
        # Load the songs and annotations
        data = 'Explain the song "{}", written by {}.\n Lyrics: {}.\n Explanation:'.format(title, artist, lyrics)
    elif prompt_type == "question_context":
        data = 'question: what is the meaning of {} in his song "{}"?\n' \
               'context: {}.\n answer:'.format(artist, title, lyrics)
    else:  # None: no prompt
        data = lyrics
    # add start token
    data = "<|startoftext|> " + data
    return data
