#!/bin/bash
python set_config.py state_int=0 data_args!mini_int=0 prompt_args!prompt_type=lyrics_meaning_with_metadata \
num_train_epochs_int=4 model_name=EleutherAI/gpt-neo-1.3B
python finetuning_script.py
python set_config.py state_int=1
python finetuning_script.py
python set_config.py state_int=2

#['lyrics_meaning', 'lyrics_meaning_with_metadata', 'song', 'song_with_metadata', 'question_context', 'question_context_with_metadata', None]
#['gpt2', 'gpt2-medium', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-2.7B']
