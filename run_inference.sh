#!/bin/bash
python set_config.py state_int=3 prompt_args!prompt_type=lyrics_meaning_with_metadata \
model_name=EleutherAI/gpt-neo-1.3B \
model_path=checkpoint_gpt2-medium_lyrics_meaning_with_metadata_2022-03-10-18-44-15
python finetuning_script.py

python set_config.py state_int=3 prompt_args!prompt_type=lyrics_meaning_with_metadata \
model_name=EleutherAI/gpt-neo-1.3B \
model_path=checkpoint_gpt2-medium_lyrics_meaning_with_metadata_2022-03-10-18-44-15
python finetuning_script.py

#['lyrics_meaning', 'lyrics_meaning_with_metadata', 'song', 'song_with_metadata', 'question_context', 'question_context_with_metadata', None]
#['gpt2', 'gpt2-medium', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-2.7B']
