#!/bin/bash
echo 'Starting set_config...'		                               			
python set_config.py					num_train_epochs_int=4	take_mini_int=0	batch_size_int=4	learning_rate_float=0.000002	add_prompt=True	prompt_type=song_metadata
echo 'Starting first run...'
python T5_vanilla_model.py
echo 'Finished first run...'
python set_config.py prompt_type=constant
python T5_vanilla_model.py
python set_config.py prompt_type=question_context
python T5_vanilla_model.py
echo 'Finished...'