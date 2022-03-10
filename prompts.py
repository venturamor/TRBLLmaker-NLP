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


def generate_prompts(lyrics, meaning, artist="artist", title="song", prompt_type=None, for_eval=False):
    if for_eval:
        meaning = ""
    else:
        meaning = " " + meaning[0]
    if prompt_type == "lyrics_meaning":
        data = "lyrics: {}. meaning:{}".format(lyrics, meaning)
    elif prompt_type == "lyrics_meaning_with_metadata":
        data = "artist: {}. title: {}. lyrics: {}. meaning:{}".format(artist, title, lyrics, meaning)
    elif prompt_type == "song":
        data = 'explain the song. lyrics: {}. meaning:{}'.format(lyrics, meaning)
    elif prompt_type == "song_with_metadata":
        # Load the songs and annotations
        data = 'explain the song "{}", written by {}. lyrics: {}. meaning:{}'.format(title, artist, lyrics, meaning)
    elif prompt_type == "question_context":
        data = 'question: what is the meaning of artist in his song? ' \
               'context: {}. answer:{}'.format(lyrics, meaning)
    elif prompt_type == "question_context_with_metadata":
        data = 'question: what is the meaning of {} in his song "{}"? ' \
               'context: {}. answer:{}'.format(artist, title, lyrics, meaning)
    else:  # None: no prompt
        data = lyrics
    # add start token
    data = "<|startoftext|> " + data
    if not for_eval:
        data = data + " <|endoftext|>"
    return data
