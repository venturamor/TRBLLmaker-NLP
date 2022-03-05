"""
Now load the data line by line
"""
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import datasets
import os
from transformers import TFGPT2LMHeadModel
from transformers import GPT2Tokenizer
from box import Box
import yaml
from config_parser import *
from tqdm import tqdm
from prompts import *


def generate_txt_for_training(test_path, train_name, eval_name, prompt_type):
    """
    This function will generate the txt file for training
    :param test_path:
    :return:
    """
    dataset_name = 'TRBLL_dataset.py'
    samples_dataset = datasets.load_dataset(dataset_name)['train']

    dataset = [generate_prompts(lyrics=row['data'], meaning=row['labels'], artist=row['artist'], title=row['title'],
                                prompt_type=prompt_type) for row in samples_dataset]

    train, eval = train_test_split(dataset, train_size=.9, random_state=21)

    with open(os.path.join(test_path, train_name + '_' + prompt_type + '.txt'), 'w+') as file_handle:
        file_handle.write("<|endoftext|>".join(train))

    with open(os.path.join(test_path, eval_name + '_' + prompt_type + '.txt'), 'w+') as file_handle:
        file_handle.write("<|endoftext|>".join(eval))


# Using the model
def evaluate_model_on_test_data(model_name, model_path):
    """
    This function will evaluate the model on the test data
    :param model_name:
    :param model_path:
    """
    # Load the data
    dataset_name = training_args.train_args.dataset_name
    samples_dataset = datasets.load_dataset(dataset_name)['test']
    # Load the model
    model = TFGPT2LMHeadModel.from_pretrained(model_path, from_pt=True)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # set a seed for reproducibility
    np.random.seed(21)

    # Evaluate the model on the test data
    random_int = np.random.randint(0, len(samples_dataset))
    input_prompt = generate_prompts(lyrics=samples_dataset[random_int]['data'][0],
                                    meaning=samples_dataset[random_int]['labels'][0],
                                    artist=samples_dataset[random_int]['artist'][0],
                                    title=samples_dataset[random_int]['title'][0],
                                    prompt_type=training_args.train_args.prompt.prompt_type)
    input_ids = tokenizer.encode(input_prompt, return_tensors='tf')

    generated_text_samples = model.generate(
        input_ids,
        max_length=training_args.eval_after_train_args.max_length,
        num_return_sequences=training_args.eval_after_train_args.num_return_sequences,
        no_repeat_ngram_size=2,
        repetition_penalty=1.5,
        top_p=0.92,
        temperature=.85,
        do_sample=True,
        top_k=125,
        early_stopping=True
    )



    #Print output for each sequence generated above
    for i, beam in enumerate(generated_text_samples):
        print("{}: {}".format(i, tokenizer.decode(beam, skip_special_tokens=True)))
        print()


if __name__ == '__main__':
    # Change flags according to your needs
    prepare_data = True  # Prepare the data - only need to do this once
    run_model = False
    # Prepare data if needed
    if prepare_data:
        data_for_finetuning_path = private_args.path.data_for_finetuning_path
        generate_txt_for_training(data_for_finetuning_path, train_name=private_args.path.train_name,
                                  eval_name=private_args.path.eval_name, prompt_type=private_args.path.prompt_type)
    # Run model if needed
    if run_model:
        model_path = private_args.path.model_path
        model_name = private_args.path.model_name
        # Evaluate model on test data - this will take a while
        # The parameters are in the config file
        evaluate_model_on_test_data(model_name, model_path)



# Run the script
#  python run_clm.py
#  --model_type gpt2-medium
#  --model_name_or_path gpt2-medium
#  --train_file <data_for_finetuning_path\train_name>
#  --do_train
#  --validation_file <data_for_finetuning_path\eval_name>
#  --do_eval
#  --per_gpu_train_batch_size 4
#  --save_steps -1
#  --num_train_epochs 1
#  --fp16
#  --output_dir=<model_name_or_path>
