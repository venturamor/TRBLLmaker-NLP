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
from evaluate_models import *

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

    with open(os.path.join(test_path, train_name + '.txt'), 'w+') as file_handle:
        file_handle.write("<|endoftext|>".join(train))

    with open(os.path.join(test_path, eval_name + '.txt'), 'w+') as file_handle:
        file_handle.write("<|endoftext|>".join(eval))


# Using the model
def evaluate_model_on_test_data(model_name, model_path, file_name, number_of_samples=10):
    """
    Evaluate model on test samples
    """
    dataset_name = training_args.train_args.dataset_name
    samples_dataset = datasets.load_dataset(dataset_name)['test']
    temperature = training_args.eval_after_train_args.temperature
    print("Loaded {} samples from {}".format(len(samples_dataset), dataset_name))

    # Set seed for reproducibility
    torch.manual_seed(21)
    np.random.seed(21)

    # choose random number_of_samples from the dataset
    samples = torch.randint(0, len(samples_dataset['data']) - 1, (number_of_samples,))

    # create a doc file to write the generated prompts
    doc = docx.Document()
    doc.add_heading('Predicted annotations by different models, prompts and temperature', 0)

    full_df = pd.DataFrame()

    # Run for each sample
    for index in samples:
        lyrics = samples_dataset['data'][index][0]
        meaning = samples_dataset['labels'][index][0]
        artist = samples_dataset['artist'][index][0]
        title = samples_dataset['title'][index][0]

        # Run for each prompt type
        prompt_type = training_args.train_args.prompt.prompt_type
        print("Generating prompts for {}".format(prompt_type))
        input_prompt = generate_prompts(lyrics, meaning, artist, title, prompt_type)
        decode_methods = ['greedy', 'beam search', 'sampling', 'top-k sampling', 'top-p sampling']
        for decode_method_index in range(len(decode_methods)):
            evaluation_df = run_inference_on_sample(model_name=model_name, input_prompt=input_prompt,
                                                    decode_method_index=decode_method_index, TF=True)
            evaluation_df['model'] = model_name
            evaluation_df['prompt_type'] = prompt_type
            full_df = full_df.append(evaluation_df)
            for i, row in evaluation_df.iterrows():
                generated, input_text = row['predicted_text'], row['input_prompt']
                # Save to docx file
                para = doc.add_paragraph("Model: {},\n prompt: {},\n temperature: {} \n\n"
                                         .format(model_name, prompt_type, temperature))
                para.add_run("lyrics: {}.\n meaning: {} \n\n".format(lyrics, meaning))
                # Print the generated prompt highlighted with green color
                para.add_run("Gerenated text:\n").font.highlight_color \
                    = docx.enum.text.WD_COLOR_INDEX.RED
                para.add_run("{} ".format(input_prompt)).font.highlight_color = \
                    docx.enum.text.WD_COLOR_INDEX.YELLOW
                para.add_run("{} \n\n\n".format(generated.split(input_prompt)[1])).font.highlight_color \
                    = docx.enum.text.WD_COLOR_INDEX.GREEN

    doc.save('{}_{}.docx'.format(file_name, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    # save df as pickle
    full_df.to_pickle('{}_{}.pkl'.format(file_name, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    print("Saved {}_{}.pkl".format(file_name, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    # Save to csv file
    full_df.to_csv('{}_{}.csv'.format(file_name, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    print("Saved {}_{}.csv".format(file_name, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    print("Done")

# # Load the data
# dataset_name = training_args.train_args.dataset_name
# samples_dataset = datasets.load_dataset(dataset_name)['test']
# # Load the model
# model = TFGPT2LMHeadModel.from_pretrained(model_path, from_pt=True)
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#
# # set a seed for reproducibility
# np.random.seed(21)
#
# # Evaluate the model on the test data
# random_int = np.random.randint(0, len(samples_dataset))
# input_prompt = generate_prompts(lyrics=samples_dataset[random_int]['data'][0],
#                                 meaning=samples_dataset[random_int]['labels'][0],
#                                 artist=samples_dataset[random_int]['artist'][0],
#                                 title=samples_dataset[random_int]['title'][0],
#                                 prompt_type=training_args.train_args.prompt.prompt_type)
# input_ids = tokenizer.encode(input_prompt, return_tensors='tf')
# generated_text_samples = model.generate(
#     input_ids,
#     max_length=training_args.eval_after_train_args.max_length,
#     num_return_sequences=training_args.eval_after_train_args.num_return_sequences,
#     no_repeat_ngram_size=2,
#     repetition_penalty=1.5,
#     top_p=0.92,
#     temperature=.85,
#     do_sample=True,
#     top_k=125,
#     early_stopping=True
# )
# #Print output for each sequence generated above
# for i, beam in enumerate(generated_text_samples):
#     print("{}: {}".format(i, tokenizer.decode(beam, skip_special_tokens=True)))
#     print()


if __name__ == '__main__':
    prepare_data = True  # Prepare the data
    run_model = False  # Run the model
    # Prepare data if needed
    if prepare_data:
        data_for_finetuning_path = private_args.path.data_for_finetuning_path
        generate_txt_for_training(data_for_finetuning_path, train_name=private_args.name.train_name,
                                  eval_name=private_args.name.eval_name,
                                  prompt_type=training_args.train_args.prompt.prompt_type)
    # Run model if needed
    if run_model:
        model_path = private_args.path.model_path
        model_name = private_args.path.model_name
        # Evaluate model on test data - this will take a while
        file_name = "predictions_after_training"
        number_of_samples = training_args.eval_after_train_args.num_samples
        evaluate_model_on_test_data(model_name, model_path, file_name, number_of_samples=number_of_samples)



# Run the script - training
#  python /home/student/mor_nlp/transformers/examples/pytorch/language-modeling/run_clm.py \
#  --model_type gpt2 \
#  --model_name_or_path gpt2 \
#  --train_file "/home/student/mor_nlp/data/tmp/train_tmp.txt" \
#  --do_train \
#  --validation_file "/home/student/mor_nlp/data/tmp/eval_tmp.txt" \
#  --do_eval \
#  --per_gpu_train_batch_size 4 \
#  --save_steps -1 \
#  --num_train_epochs 4 \
#  --fp16 \
#  --output_dir="/home/student/mor_nlp/checkpoints3"
