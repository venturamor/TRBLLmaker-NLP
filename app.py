import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import datasets
import os
from transformers import TFGPT2LMHeadModel
from transformers import GPT2Tokenizer
from box import Box
import yaml
from tqdm import tqdm
from prompts import *
from evaluate_models import *
from config_parser import *
from set_config import set_specific_config
from finetuning_script import decode_fn
if __name__ == '__main__':
    print("Welcome to the GPT-2 song meaning predictor!")
    # Load config variables
    main_path = private_args.path.main_path
    results_path = training_args.path_args.results_path
    model_path = "checkpoint_song_with_metadata_bs_64_2022-03-11-14-25-07"
    file_path = os.path.join(main_path, results_path, model_path)
    model_name = training_args.train_args.model_name

    # Load the model
    model = GPT2LMHeadModel.from_pretrained(file_path)

    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token  # to avoid an error

    # Wait for the user input
    while True:
        print("Insert a line from a song: ")
        line = input()
        if line == "":
            print("Insert a line from a song")
        else:
            print("You inserted: " + line)
            print("calculating...")
            input_prompt = generate_prompts(lyrics=line, meaning="", artist="artist",
                                            title="title", prompt_type="question_context_with_metadata",
                                            for_eval=True)
            outputs = decode_fn(model=model, tokenizer=tokenizer, input_prompt=input_prompt,
                                decode_method='beam search', temperature=0.9,
                                num_return_sequences=1, max_input_length=256)

            pred_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            print("Predicted text: {}\n".format(pred_text))
            print("\n")
            print("\n")
            print("\n")
