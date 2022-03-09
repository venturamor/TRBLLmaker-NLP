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
from tqdm import tqdm
from prompts import *
from evaluate_models import *
from config_parser import *

def decode_fn(model, tokenizer, input_prompt, decode_method, TF, temperature, num_return_sequences):
    # additional parameters for better decoding - repetition_penalty, min_length
    # encode prompt
    if TF:
        input_ids = tokenizer.encode(input_prompt, return_tensors='tf')
    else:
        input_ids = tokenizer(input_prompt, return_tensors="pt").input_
    # predict
    if decode_method == 'greedy':
        # greedy
        outputs = model.generate(input_ids, max_length=training_args.eval_pretrained_args.max_length,
                                 temperature=temperature, num_return_sequences=1,
                                 )
    elif decode_method == 'beam search':
        # beam search with penalty on repeat
        outputs = model.generate(input_ids, max_length=training_args.eval_pretrained_args.max_length,
                                 num_beams=3, early_stopping=True,
                                 num_return_sequences=num_return_sequences,
                                 no_repeat_ngram_size=2
                                 )
    elif decode_method == 'sampling':
        # sampling
        outputs = model.generate(input_ids, do_sample=True, top_k=0,
                                 max_length=training_args.eval_pretrained_args.max_length,
                                 num_return_sequences=num_return_sequences,
                                 temperature=temperature
                                 )
    elif decode_method == 'top-k sampling':
        # top-k sampling
        outputs = model.generate(input_ids, do_sample=True, top_k=50,
                                 max_length=training_args.eval_pretrained_args.max_length,
                                 num_return_sequences=num_return_sequences,
                                 )
    else:  # decode_method == 'top-p sampling' (default)
        # top-p sampling
        outputs = model.generate(input_ids, do_sample=True, top_k=0, top_p=0.92,
                                 max_length=training_args.eval_pretrained_args.max_length,
                                 num_return_sequences=num_return_sequences,
                                 )
    return outputs


def generate_txt_for_training(test_path, train_name, eval_name, prompt_type):
    """
    This function will generate the txt file for training
    :param test_path:
    :return:
    """
    dataset_name = 'TRBLL_dataset.py'
    samples_dataset_train = datasets.load_dataset(dataset_name)['train']
    samples_dataset_validation = datasets.load_dataset(dataset_name)['validation']

    train_dataset = [generate_prompts(lyrics=row['data'], meaning=row['labels'], artist=row['artist'], title=row['title'],
                                prompt_type=prompt_type) for row in samples_dataset_train]

    validation_dataset = [generate_prompts(lyrics=row['data'], meaning=row['labels'], artist=row['artist'],
                                           title=row['title'], prompt_type=prompt_type)
                          for row in samples_dataset_validation]

    with open(os.path.join(test_path, train_name + '.txt'), 'w+') as file_handle:
        file_handle.write("<|endoftext|>".join(train_dataset))

    with open(os.path.join(test_path, eval_name + '.txt'), 'w+') as file_handle:
        file_handle.write("<|endoftext|>".join(validation_dataset))

    print("Generated txt files for training")


# Using the model
def evaluate_model_on_test_data(model_name, model_path, file_name, number_of_samples=10, after_training=False):
    """
    Evaluate model on test samples
    """

    if after_training:
        TF = True  # Currently using TensorFlow for evaluation of the model (not PyTorch)
        model_names = [model_name]
        model_paths = [model_path]
    else:
        TF = False
        model_names = ['gpt2', 'gpt2-medium', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-2.7B']
        model_paths = ['gpt2', 'gpt2-medium', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-2.7B']

    # Load the test data
    dataset_name = training_args.train_args.dataset_name
    samples_dataset = datasets.load_dataset(dataset_name)['test']
    temperature = training_args.eval_after_train_args.temperature
    print("Loaded {} samples from {}".format(len(samples_dataset), dataset_name))
    num_return_sequences = training_args.eval_after_train_args.num_return_sequences
    # Set seed for reproducibility
    torch.manual_seed(21)
    np.random.seed(21)

    # choose random number_of_samples from the dataset
    samples = torch.randint(0, len(samples_dataset['data']) - 1, (number_of_samples,))

    # create a doc file to write the generated prompts
    doc = docx.Document()
    headline = "after training" if after_training else "before training"
    doc.add_heading('Predicted annotation {}'.format(headline), 0)

    # Dataframe to store the results
    full_df = pd.DataFrame(columns=['input_prompt', 'predicted_text', 'decode_method', 'temperature',
                                    'model', 'prompt_type'])

    # Loop over the models
    for model_name, model_path in tqdm(zip(model_names, model_paths)):
        print("Run for each model.\nCurrent model: {}".format(model_name))
        #  Load tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # Load the model
        if model_name == 'gpt2' or model_name =='gpt2-medium':
            if TF:
                model = TFGPT2LMHeadModel.from_pretrained(model_path)
            else:
                model = GPT2LMHeadModel.from_pretrained(model_path)
        else:
            model = GPTNeoForCausalLM.from_pretrained(model_path)

        df_inference = pd.DataFrame(columns=['example_index', 'input_prompt', 'predicted_text', 'decode_method',
                                             'temperature', 'model', 'prompt_type', 'meaning'])

        # Run for each sample
        for index in tqdm(samples):
            print("Run for each sample...")
            lyrics = samples_dataset['data'][index][0]
            meaning = samples_dataset['labels'][index][0]
            artist = samples_dataset['artist'][index][0]
            title = samples_dataset['title'][index][0]

            # Run for each prompt type
            prompt_type = training_args.train_args.prompt.prompt_type
            print("Generating prompts for {}".format(prompt_type))
            input_prompt = generate_prompts(lyrics=lyrics, meaning=meaning, artist=artist, title=title,
                                            prompt_type=prompt_type, for_eval=True)

            decode_methods = ['greedy', 'beam search', 'sampling', 'top-k sampling', 'top-p sampling']
            for decode_method in decode_methods:
                outputs = decode_fn(model=model, tokenizer=tokenizer, input_prompt=input_prompt, decode_method=decode_method,
                          TF=TF, temperature=temperature, num_return_sequences=num_return_sequences)

                # decode
                pred_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                input_list, generated_text_list, index_list = [], [], []
                decode_method_list, temperature_list = [], []

                for pred in pred_text:
                    input_list.append(input_prompt)
                    index_list.append(index.tolist())
                    decode_method_list.append(decode_method)
                    temperature_list.append(temperature)
                    pred_splitted = pred.split(input_prompt)
                    if len(pred_splitted) <= 1:
                        pred = "Empty"
                    elif len(pred_splitted) == 2:
                        pred = pred.split(input_prompt)[1]
                    else:
                        pred = "More than one repetition: " + pred
                    generated_text_list.append(pred)
                df_curr = pd.DataFrame({'example_index': index_list, 'input_prompt': input_list,
                                        'predicted_text': generated_text_list, 'decode_method': decode_method_list,
                                        'temperature': temperature_list, 'model': model_name, 'prompt_type': prompt_type,
                                        'meaning': meaning})
                df_inference = pd.concat([df_inference, df_curr], ignore_index=True)

    # save inference results
    for i, row in df_inference.iterrows():
        generated, input_text = row['predicted_text'], row['input_prompt']
        model_name, prompt_type, temperature = row['model'], row['prompt_type'], row['temperature']
        meaning = row['meaning']

        generated = "EMPTY" if len(generated.split(input_text)) <= 1 else generated.split(input_text)[1]
        # Save to docx file
        para = doc.add_paragraph("Model: {},\n prompt: {},\n temperature: {} \n\n"
                                 .format(model_name, prompt_type, temperature))
        para.add_run("lyrics: {}.\n meaning: {} \n\n".format(input_text, meaning))
        # Print the generated prompt highlighted with green color
        para.add_run("Gerenated text:\n").font.highlight_color \
            = docx.enum.text.WD_COLOR_INDEX.RED
        para.add_run("{} \n\n\n".format(generated)).font.highlight_color \
            = docx.enum.text.WD_COLOR_INDEX.GREEN


    doc.save('{}_{}.docx'.format(file_name, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    # save df as pickle
    full_df.to_pickle('{}_{}.pkl'.format(file_name, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    print("Saved {}_{}.pkl".format(file_name, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    # Save to csv file
    full_df.to_csv('{}_{}.csv'.format(file_name, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    print("Saved {}_{}.csv".format(file_name, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    print("Done")


if __name__ == '__main__':
    states = ["prepare_data", "train", "eval", "eval_pretrained"]
    state = 0
    curr_state = states[state]
    main_path = private_args.path.main_path
    # Prepare the data
    if curr_state == "prepare_data":
        data_for_finetuning_path = os.path.join(main_path, private_args.path.data_for_finetuning_path)
        generate_txt_for_training(data_for_finetuning_path, train_name=private_args.name.train_name,
                                  eval_name=private_args.name.eval_name,
                                  prompt_type=training_args.train_args.prompt.prompt_type)
    elif curr_state == "train":
        # Run the mode
        model_type = 'gpt2-medium'
        model_name_or_path = 'gpt2-medium'
        model_path = private_args.path.model_path
        train_file = os.path.join(main_path, private_args.path.data_for_finetuning_path, private_args.name.train_name)
        validation_file = os.path.join(main_path, private_args.path.data_for_finetuning_path, private_args.name.eval_name)
        output_dir = os.path.join(main_path, private_args.path.output_dir, model_path)
        batch_size = training_args.train_args.batch_size
        num_train_epochs = training_args.train_args.num_train_epochs
        training_script = os.path.join(main_path, 'transformers/examples/pytorch/language-modeling/run_clm.py')

        print("Run the following command to see the results:")
        print("nohup python {} --model_type {} --model_name_or_path {} --train_file {} --do_train --validation_file {}"
              " --do_eval --per_gpu_train_batch_size {} --save_steps -1 --num_train_epochs {} --fp16 --output_dir {}"
              " --overwrite_output_dir & tail -f nohup.out".format(training_script, model_type, model_name_or_path, train_file + '.txt',
                validation_file + '.txt', batch_size, num_train_epochs, output_dir))
    if curr_state == "eval":
        model_path = private_args.path.model_path
        model_name = private_args.name.model_name
        # Evaluate model on test data - this will take a while
        results_path = private_args.path.results_path
        after_training_folder = private_args.path.after_training_folder
        file_name = "predictions_after_training"
        file_path = os.path.join(main_path, results_path, after_training_folder, file_name)
        number_of_samples = training_args.eval_after_train_args.num_samples
        evaluate_model_on_test_data(model_name, model_path, file_path, number_of_samples=number_of_samples,
                                    after_training=True)

    elif curr_state == "eval_pretrained":
        number_of_samples = training_args.eval_pretrained_args.num_samples
        main_path = private_args.path.main_path
        results_path = private_args.path.results_path
        pretraining_folder = private_args.path.pretraining_folder
        file_name = "predictions_before_training"
        file_path = os.path.join(main_path, results_path, pretraining_folder, file_name)
        evaluate_model_on_test_data("", "", file_path, number_of_samples=number_of_samples,
                                    after_training=False)
