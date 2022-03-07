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

def generate_txt_for_training(test_path):
    """
    This function will generate the txt file for training
    :param test_path:
    :return:
    """
    dataset_name = 'TRBLL_dataset.py'
    samples_dataset = datasets.load_dataset(dataset_name)['train']

    dataset = ["<|title|>" + row['data'][0].strip() + ' ' + row['labels'][0].strip() for row in samples_dataset]

    train, eval = train_test_split(dataset, train_size=.9, random_state=2020)

    with open(os.path.join(test_path, 'train_tmp.txt'), 'w+') as file_handle:
      file_handle.write("<|endoftext|>".join(train))

    with open(os.path.join(test_path, 'eval_tmp.txt'), 'w+') as file_handle:
      file_handle.write("<|endoftext|>".join(eval))


# Using the model
def evaluate_model_on_test_data(model_name_or_path):
    """
    This function will evaluate the model on the test data
    :param test_path:
    :param model_name_or_path:
    :param tokenizer_name_or_path:
    :param model_type:
    :return:
    """
    dataset_name = 'TRBLL_dataset.py'
    samples_dataset = datasets.load_dataset(dataset_name)['train']

    model = TFGPT2LMHeadModel.from_pretrained("/home/tok/TRBLLmaker/transformers/examples/pytorch/language-modeling/output_dir", from_pt=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    random_int = np.random.randint(0, len(samples_dataset))
    input_prompt = "<|title|>" + samples_dataset[random_int]['data'][0]
    input_ids = tokenizer.encode(input_prompt, return_tensors='tf')

    generated_text_samples = model.generate(
        input_ids,
        max_length=150,
        num_return_sequences=1,
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
  # prepare
  # data_path = '/home/student/mor_nlp/data/tmp'
  # generate_txt_for_training(data_path)

  # eval
  model_name_or_path = '/home/tok/TRBLLmaker/checkpoints2'
  evaluate_model_on_test_data(model_name_or_path)



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
