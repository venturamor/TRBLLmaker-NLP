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
from set_config import set_specific_config


def decode_fn(model, tokenizer, input_prompt, decode_method, temperature, num_return_sequences, max_input_length):
    # additional parameters for better decoding - repetition_penalty, min_length
    # encode prompt
    #if TF:
    #    input_ids = tokenizer.encode(input_prompt, return_tensors='tf')
    #else:
    input_ids = tokenizer(input_prompt, return_tensors="pt",
                          max_length=max_input_length, truncation=True, padding=True).input_ids
    # predict
    if decode_method == 'greedy':
        # greedy
        outputs = model.generate(input_ids, max_length=training_args.eval_args.max_length,
                                 temperature=temperature, num_return_sequences=1,
                                 )
    elif decode_method == 'beam search':
        # beam search with penalty on repeat
        outputs = model.generate(input_ids, max_length=training_args.eval_args.max_length,
                                 num_beams=3, early_stopping=True,
                                 num_return_sequences=num_return_sequences,
                                 no_repeat_ngram_size=2
                                 )
    elif decode_method == 'sampling':
        # sampling
        outputs = model.generate(input_ids, do_sample=True, top_k=0,
                                 max_length=training_args.eval_args.max_length,
                                 num_return_sequences=num_return_sequences,
                                 temperature=temperature
                                 )
    elif decode_method == 'top-k sampling':
        # top-k sampling
        outputs = model.generate(input_ids, do_sample=True, top_k=50,
                                 max_length=training_args.eval_args.max_length,
                                 num_return_sequences=num_return_sequences,
                                 )
    else:  # decode_method == 'top-p sampling' (default)
        # top-p sampling
        outputs = model.generate(input_ids, do_sample=True, top_k=0, top_p=0.92,
                                 max_length=training_args.eval_args.max_length,
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

    train_dataset = [generate_prompts(lyrics=row['data'][0], meaning=row['labels'][0], artist=row['artist'][0],
                                      title=row['title'][0], prompt_type=prompt_type)
                     for row in samples_dataset_train]

    # validation_dataset = [generate_prompts(lyrics=row['data'], meaning=row['labels'], artist=row['artist'],
    #                                        title=row['title'], prompt_type=prompt_type)
    #                       for row in samples_dataset_validation]

    # split the dataset into train and validation
    train_dataset, validation_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)



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
        # TF = True  # Currently using TensorFlow for evaluation of the model (not PyTorch)
        model_names = [model_name]
        model_paths = [model_path]
    else:
        # TF = False
        model_names = ['gpt2', 'gpt2-medium', 'EleutherAI/gpt-neo-1.3B']
        model_paths = ['gpt2', 'gpt2-medium', 'EleutherAI/gpt-neo-1.3B']

    # Load the test data
    max_input_length = training_args.eval_args.max_input_length
    dataset_name = training_args.data_args.dataset_name
    samples_dataset = datasets.load_dataset(dataset_name)['test']
    temperature = training_args.eval_args.temperature
    print("Loaded {} samples from {}".format(len(samples_dataset), dataset_name))
    num_return_sequences = training_args.eval_args.num_return_sequences
    # Set seed for reproducibility
    torch.manual_seed(21)
    np.random.seed(21)

    # take only a subset of the test data
    all_test_data, samples = train_test_split(samples_dataset, test_size=1, random_state=5)

    # create a doc file to write the generated prompts
    doc = docx.Document()
    headline = "after training" if after_training else "before training"
    doc.add_heading('Predicted annotation {}'.format(headline), 0)

    # Dataframe to store the results
    df_inference = pd.DataFrame(columns=['example_index', 'input_prompt', 'predicted_text', 'decode_method',
                                         'temperature', 'model', 'prompt_type', 'gt_meaning'])

    # Loop over the models
    for model_name, model_path in tqdm(zip(model_names, model_paths)):
        print("Run for each model.\nCurrent model: {}".format(model_name))
        #  Load tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # when generating, we will use the logits of right-most token to predict the next token
        # so the padding should be on the left
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token  # to avoid an error
        # Load the model
        if model_name == 'gpt2' or model_name =='gpt2-medium':
            #if TF:
            #    model = TFGPT2LMHeadModel.from_pretrained(model_path, from_pt=True)
            #else:
            model = GPT2LMHeadModel.from_pretrained(model_path)
        else:
            model = GPTNeoForCausalLM.from_pretrained(model_path)

        # def preprocess_function(samples):
        #     output = tokenizer(samples["data"])
        #     return output
        #
        # # Run for each sample
        # tokenized_datasets = samples_dataset.map(
        #     preprocess_function,
        #     batched=True,
        #     remove_columns=samples_dataset.column_names,
        # )
        # print(tokenized_datasets.column_names)

        # model_name = 'gpt2'
        # dataset_name = training_args.data_args.dataset_name
        # samples_dataset = datasets.load_dataset(dataset_name)['test']
        # tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        #
        # data = [data[0] for data in samples_dataset['data']]
        # tokenized_data = tokenizer(data)
        # model = GPT2LMHeadModel.from_pretrained(model_name)
        # data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        # colladed_data = data_collator(data)
        # print(colladed_data['input_ids'].shape)
        #
        #
        # print("Generating prompts...")

        for index, (lyrics, meaning, artist, title) in \
                tqdm(enumerate(zip(samples['data'], samples['labels'], samples['artist'], samples['title']))):

            prompt_type = training_args.prompt_args.prompt_type
            # Run for each prompt type
            if after_training:
                prompt_types = [prompt_type]
            else:
                prompt_types = ['lyrics_meaning', 'lyrics_meaning_with_metadata', 'song', 'song_with_metadata',
                                'question_context', 'question_context_with_metadata', None]
            for prompt_type in tqdm(prompt_types):
                print("Generating prompts for {}".format(prompt_type))
                input_prompt = generate_prompts(lyrics=lyrics[0], meaning=meaning[0], artist=artist[0],
                                                title=title[0], prompt_type=prompt_type,
                                                for_eval=True)

                decode_methods = ['greedy', 'beam search', 'sampling', 'top-k sampling', 'top-p sampling']
                for decode_method in decode_methods:
                    outputs = decode_fn(model=model, tokenizer=tokenizer, input_prompt=input_prompt,
                                        decode_method=decode_method, temperature=temperature,
                                        num_return_sequences=num_return_sequences, max_input_length=max_input_length)

                    # decode
                    pred_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                    input_list, generated_text_list, predicted_meaning_list, index_list = [], [], [], []
                    decode_method_list, temperature_list = [], []

                    for pred in pred_text:
                        input_list.append(input_prompt)
                        index_list.append(index)
                        generated_text_list.append(pred)
                        decode_method_list.append(decode_method)
                        temperature_list.append(temperature)
                        pred_splitted = pred.split(input_prompt)
                        if len(pred_splitted) <= 1:
                            pred = "Empty"
                        elif len(pred_splitted) == 2:
                            pred = pred_splitted[1]
                        else:
                            pred = "More than one repetition: " + pred
                        predicted_meaning_list.append(pred)
                    df_curr = pd.DataFrame({'example_index': index_list,
                                            'input_prompt': input_list,
                                            'predicted_text': generated_text_list,
                                            'predicted_meaning': predicted_meaning_list,
                                            'decode_method': decode_method_list,
                                            'temperature': temperature_list,
                                            'model': model_name,
                                            'prompt_type': prompt_type,
                                            'gt_meaning': meaning, 'lyrics': lyrics, 'artist': artist, 'title': title
                                            })
                    df_inference = pd.concat([df_inference, df_curr], ignore_index=True)

    # save inference results
    for i, row in df_inference.iterrows():
        generated, input_text = row['predicted_text'], row['input_prompt']
        model_name, prompt_type, temperature = row['model'], row['prompt_type'], row['temperature']
        meaning = row['gt_meaning']

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
    df_inference.to_pickle('{}_{}.pkl'.format(file_name, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    print("Saved {}_{}.pkl".format(file_name, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    # Save to csv file
    df_inference.to_csv('{}_{}.csv'.format(file_name, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    print("Saved {}_{}.csv".format(file_name, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    print("Done")


if __name__ == '__main__':
    states = ["prepare_data", "train", "eval", "eval_pretrained"]
    state = training_args.train_args.state
    curr_state = states[state]
    main_path = private_args.path.main_path
    prompt_type = training_args.prompt_args.prompt_type
    # Prepare the data
    if curr_state == "prepare_data":
        print("Preparing data...")
        data_for_finetuning_path = os.path.join(main_path, training_args.path_args.data_for_finetuning_path)
        generate_txt_for_training(data_for_finetuning_path, train_name=training_args.path_args.train_name,
                                  eval_name=training_args.path_args.eval_name,
                                  prompt_type=prompt_type)
    elif curr_state == "train":
        print("preparing args for training...")
        # Run the mode
        model_type = training_args.train_args.model_name
        model_name_or_path = model_type
        gradient_accumulation_steps = training_args.train_args.gradient_accumulation_steps
        model_path = "checkpoint_{}_bs_{}_{}".format(prompt_type, gradient_accumulation_steps,
                                                  datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        # Save details for future use to models.txt file (if not exist, create it)
        if not os.path.exists(os.path.join(main_path, "models.txt")):
            with open(os.path.join(main_path, "models.txt"), "w") as f:
                f.write("model_name: {}, model_path: {}, prompt_type: {}"
                        "\n".format(model_type, model_path, prompt_type))

        # Set model_path for evaluation after training
        set_specific_config("private_config.yaml", "path", "model_path", model_path)
        train_file = os.path.join(main_path, training_args.path_args.data_for_finetuning_path,
                                  training_args.path_args.train_name)
        validation_file = os.path.join(main_path, training_args.path_args.data_for_finetuning_path,
                                       training_args.path_args.eval_name)
        output_dir = os.path.join(main_path, training_args.path_args.output_dir, model_path)
        batch_size = training_args.train_args.batch_size
        num_train_epochs = training_args.train_args.num_train_epochs
        training_script = os.path.join(main_path, 'transformers/examples/pytorch/language-modeling/run_clm.py')
        eval_steps = training_args.train_args.eval_steps
        logging_steps = eval_steps
        train_command = "nohup python {} --model_type {} --model_name_or_path {} --train_file {} --do_train " \
                        "--validation_file {} --do_eval --per_gpu_train_batch_size {} --save_steps -1 " \
                        "--num_train_epochs {} --fp16 --output_dir {} --overwrite_output_dir " \
                        "--gradient_accumulation_steps {} --evaluation_strategy {} --eval_steps {} " \
                        "--logging_strategy {} --logging_steps {} " \
                        "& tail -f nohup.out".format(training_script, model_type, model_name_or_path,
                                                     train_file + '.txt', validation_file + '.txt', batch_size,
                                                     num_train_epochs, output_dir, gradient_accumulation_steps,
                                                     "steps", eval_steps, "steps", logging_steps)

        print("Run the following command to train:")
        print(train_command)

    if curr_state == "eval":
        print("evaluating...")
        model_path = private_args.path.model_path
        output_dir = training_args.path_args.output_dir
        model_path = os.path.join(main_path, output_dir, model_path)
        model_name = training_args.train_args.model_name
        # Evaluate model on test data - this will take a while
        results_path = training_args.path_args.results_path
        after_training_folder = training_args.path_args.after_training_folder
        file_name = "predictions_after_training"
        file_path = os.path.join(main_path, results_path, after_training_folder, file_name)
        number_of_samples = training_args.eval_args.num_samples
        prompt_type = training_args.prompt_args.prompt_type
        print("Model: {}".format(model_name))
        print("Model path: {}".format(model_path))
        print("Prompt type: {}".format(prompt_type))
        evaluate_model_on_test_data(model_name, model_path, file_path, number_of_samples=number_of_samples,
                                    after_training=True)

    elif curr_state == "eval_pretrained":
        print("evaluating pretrained model...")
        number_of_samples = training_args.eval_args.num_samples
        main_path = private_args.path.main_path
        results_path = training_args.path_args.results_path
        pretraining_folder = training_args.path_args.pretraining_folder
        file_name = "predictions_before_training"
        file_path = os.path.join(main_path, results_path, pretraining_folder, file_name)
        evaluate_model_on_test_data("", "", file_path, number_of_samples=1,
                                    after_training=False)


#####
# from itertools import chain
# block_size = 1024
#
#
# model_name = 'gpt2'
# dataset_name = training_args.data_args.dataset_name
# raw_datasets = datasets.load_dataset(dataset_name)['test']
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#
# def group_texts(examples):
#     # Concatenate all texts.
#     concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
#     total_length = len(concatenated_examples[list(examples.keys())[0]])
#     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
#     # customize this part to your needs.
#     if total_length >= block_size:
#         total_length = (total_length // block_size) * block_size
#     # Split by chunks of max_len.
#     result = {
#         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
#         for k, t in concatenated_examples.items()
#     }
#     result["labels"] = result["input_ids"].copy()
#     return result
#
# def tokenize_function(examples):
#     data = [d[0] for d in examples['data']]
#     output = tokenizer(data)
#     # clm input could be much much longer than block_size
#     return output
#
#
# tokenized_datasets = raw_datasets.map(
#     tokenize_function,
#     batched=True,
#     desc="Running tokenizer on dataset",
# )
#
#
# lm_datasets = tokenized_datasets.map(
#     group_texts,
#     batched=True,
#     desc=f"Grouping texts in chunks of {block_size}",
# )
#
# print(lm_datasets)

# Run the script - training
#  python /home/student/mor_nlp/transformers/examples/pytorch/language-modeling/run_clm.py \
#  --model_type gpt2 \
#  --model_name_or_path gpt2 \
#  --train_file "/home/student/mor_nlp/data/tmp/train_tmp.txt" \
#  --do_train \
#  --validation_file "/home/student/mor_nlp/data/tmp/eval_tmp.txt" \
#  --do_eval \
#  --per_gpu_train_batch_size 1 \
#  --save_steps -1 \
#  --num_train_epochs 4 \
#  --fp16 \
#  --output_dir="/home/student/mor_nlp/checkpoints3" \
#  --overwrite_output_dir

