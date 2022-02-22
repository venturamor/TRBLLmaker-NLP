#!/bin/bash
#	arguments: 											dataset_name	 					model_name				num_train_epochs		batch_size		learning_rate
echo 'Starting set_config...'		                               			
python set_config.py 							TRBLL_dataset_mini.py		t5-small						4										16						0.000001
echo 'Starting first run...'
python T5_vanilla_model.py
echo 'Finished first run...'
python set_config.py 							TRBLL_dataset.py		    t5-small						4										16						0.000001
python T5_vanilla_model.py
python set_config.py 							TRBLL_dataset.py				t5-small						8										16						0.0000008
python T5_vanilla_model.py
python set_config.py 	  					TRBLL_dataset.py				t5-small						16									16						0.0000004
python T5_vanilla_model.py
python set_config.py 							TRBLL_dataset.py				t5-small						32									16						0.0000002
python T5_vanilla_model.py
python set_config.py	  					TRBLL_dataset.py				t5-small						5										16						0.0000001
python T5_vanilla_model.py
echo 'Finished...'