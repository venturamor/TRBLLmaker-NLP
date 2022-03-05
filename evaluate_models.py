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


def run_inference_on_sample(model_name, input_prompt, decode_method_index=1, temprature=0.9, num_return_sequence=1,
                            TF=False):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if model_name == 'gpt2' or model_name =='gpt2-medium':
        if TF:
            model = TFGPT2LMHeadModel.from_pretrained(model_name)
        else:
            model = GPT2LMHeadModel.from_pretrained(model_name)
    else:
        model = GPTNeoForCausalLM.from_pretrained(model_name)
    print("Model: {} loaded".format(model_name))

    decode_methods = ['greedy', 'beam search', 'sampling', 'top-k sampling', 'top-p sampling']
    decode_method = decode_methods[decode_method_index]

    # encode prompt
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

    # predict
    if decode_method == 'greedy':
        # greedy
        outputs = model.generate(input_ids, max_length=training_args.eval_pretrained_args.max_length,
                                 temprature=temprature, num_return_sequence=num_return_sequence,
                                 num_return_sequences=1,
                                 )
    elif decode_method == 'beam search':
        # beam search with penalty on repeat
        outputs = model.generate(input_ids, max_length=training_args.eval_pretrained_args.max_length,
                                 num_beams=3, early_stopping=True,
                                 num_return_sequences=training_args.eval_pretrained_args.num_return_sequences,
                                 no_repeat_ngram_size=2
                                 )
    elif decode_method == 'sampling':
        # sampling
        outputs = model.generate(input_ids, do_sample=True, top_k=0,
                                 max_length=training_args.eval_pretrained_args.max_length,
                                 num_return_sequences=training_args.eval_pretrained_args.num_return_sequences,
                                 temprature=temprature
                                 )
    elif decode_method == 'top-k sampling':
        # top-k sampling
        outputs = model.generate(input_ids, do_sample=True, top_k=50,
                                 max_length=training_args.eval_pretrained_args.max_length,
                                 num_return_sequences=training_args.eval_pretrained_args.num_return_sequences,
                                 )
    else:  # decode_method == 'top-p sampling' (default)
        # top-p sampling
        outputs = model.generate(input_ids, do_sample=True, top_k=0, top_p=0.92,
                                 max_length=training_args.eval_pretrained_args.max_length,
                                 num_return_sequences=training_args.eval_pretrained_args.num_return_sequences,
                                 )
    # additional parameters for better decoding - repetition_penalty, min_length
    # decode
    pred_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    input_list, generated_text = [], []

    for pred in pred_text:
        input_list.append(input_prompt)
        generated_text.append(pred)

    df_inference = pd.DataFrame({'original_text': input_list, 'predicted_text': generated_text})
    return df_inference


def compare_models(tf=False):
    models_names = ['gpt2', 'gpt2-medium', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-2.7B']
    prompt_types = ['lyrics_meaning', 'song_metadata', 'question_context']

    # Variables
    max_input_length = 512
    max_target_length = 512
    temperature = 0.9
    N = 2
    num_return_sequences = 2
    torch.manual_seed(21)

    #  Run for each model
    for model_name in models_names:
        # load datasets
        dataset_name = 'TRBLL_dataset.py'
        samples_dataset = datasets.load_dataset(dataset_name)['test']
        print("Loaded {} samples from {}".format(len(samples_dataset), dataset_name))

        # choose random N samples from the dataset
        samples = torch.randint(0, len(samples_dataset['data']) - 1, (N,))

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
            for prompt_type in prompt_types:
                print("Generating prompts for {}".format(prompt_type))
                input_prompt = generate_prompts(lyrics, meaning, artist, title, prompt_type)
                decode_methods = ['greedy', 'beam search', 'sampling', 'top-k sampling', 'top-p sampling']
                for decode_method_index in range(len(decode_methods)):
                    evaluation_df = run_inference_on_sample(model_name=model_name, input_prompt=input_prompt,
                                                            decode_method_index=decode_method_index, TF=tf)
                    evaluation_df['model'] = model_name
                    evaluation_df['prompt_type'] = prompt_type
                    evaluation_df['temperature'] = temperature
                    evaluation_df['decode_method'] = decode_methods[decode_method_index]
                    full_df = full_df.append(evaluation_df)

                    for i, row in evaluation_df.iterrows():
                        generated, input_text = row['predicted_text'], row['original_text']
                        # Save to docx file
                        para = doc.add_paragraph("Model: {},\n prompt: {},\n temperature: {} \n\n"
                                                 .format(model_name, prompt_type, temperature))
                        para.add_run("lyrics: {}.\n meaning: {} \n\n".format(lyrics, meaning))
                        # Print the generated prompt highlighted with green color
                        para.add_run("Gerenated text:\n").font.highlight_color = docx.enum.text.WD_COLOR_INDEX.RED
                        para.add_run("{} \n\n\n".format(generated)).font.highlight_color = docx.enum.text.WD_COLOR_INDEX.GREEN
        doc.save    ('predictions_before_training_{}.docx'.format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))

        # Save to csv file
        full_df.to_csv('predictions_before_training_{}.csv'.format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
        print("Saved predictions to csv file")
        print("Predictions saved to file")
        print("Done")


if __name__ == '__main__':
    compare_models()
