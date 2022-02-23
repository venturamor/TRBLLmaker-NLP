# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This is a dataset of songs created for TRBLLmaker"""

import os
from os.path import join
import sys
import pandas as pd
import datasets
from config_parser import config_args
from datasets import ClassLabel, Value
import yaml

# You can copy an official description
_DESCRIPTION = """\
"""

_HOMEPAGE = ""

_LICENSE = ""

# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "trbll_dataset": "",
}


class TRBLLDataset(datasets.GeneratorBasedBuilder):
    """This is a dataset of songs in hebrew with labels for metaphors"""
    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="trbll_dataset", version=VERSION, description="TRBLL dataset"),
    ]

    DEFAULT_CONFIG_NAME = "trbll_dataset"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        features = datasets.Features(
            {
                "data": datasets.Sequence(datasets.Value("string")),
                "labels": datasets.Sequence(datasets.Value("string")),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
        )

    def _split_generators(self, dl_manager):

        data_dir = config_args["train_args"]["data_dir"]
        data_types = config_args["train_args"]["data_type"]
        data_type = config_args["train_args"]["specific_type"]
        take_mini = config_args["train_args"]["take_mini"]
        str_parts = config_args["train_args"]["parts"]
        if take_mini != 0:
            str_parts = [part + '_mini.json' for part in str_parts]
        else:
            str_parts = [part + '.json' for part in str_parts]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, data_types[data_type], str_parts[0]),
                    "split": 'train',
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, data_types[data_type], str_parts[1]),
                    "split": "test",

                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, data_types[data_type], str_parts[2]),
                    "split": "validation",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        df = pd.read_json(filepath)
        data_col = config_args["train_args"]["data_col"]
        label_col = config_args["train_args"]["label_col"]
        for index, row in df.iterrows():
            yield index, {
                "data": [row[data_col]],
                "labels": [row[label_col]],
            }


def change_yml_for_dataset(config_args, specific_type: int, data_col: str, label_col: str):
    new_config_args = config_args.copy()
    new_config_args['train_args']['specific_type'] = specific_type
    new_config_args['train_args']['data_col'] = data_col
    new_config_args['train_args']['label_col'] = label_col
    file = open('config.yaml', "w")
    yaml.dump(new_config_args, file)
    file.close()
    from config_parser import config_args


if __name__ == '__main__':
    # https: // huggingface.co / docs / datasets / processing.html

    # Uncomment: to change the yml to create different dataset!:
    # change_yml_for_dataset(config_args=config_args, specific_type=1,
    #                        data_col='text', label_col='annotation')

    samples_dataset = datasets.load_dataset('TRBLL_dataset.py')
    print('done! ')
