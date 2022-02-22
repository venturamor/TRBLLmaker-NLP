#!/bin/bash
#	arguments: 											dataset_name	 					model_name				num_train_epochs		batch_size		learning_rate
echo 'Starting set_config...'		                               			
python set_config.py 							TRBLL_dataset_mini.py		t5-small					1										8							0.0000001
echo 'Starting first run...'		 
python run.py
echo 'Finished first run...'
python set_config.py 							TRBLL_dataset.py				t5-base						5										16						0.0000001
python run.py
python set_config.py 	  					TRBLL_dataset.py				t5-base						5										16						0.0000001
python run.py
python set_config.py 							TRBLL_dataset.py				t5-base						5										16						0.0000001
python run.py
python set_config.py	  					TRBLL_dataset.py				t5-base						5										32						0.0000001
python run.py
echo 'Finished...'


