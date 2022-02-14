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
"""This is a dataset of songs from the genius website"""


import os
import datasets
import pandas as pd
import yaml

# You can copy an official description
_DESCRIPTION = """\

"""

_HOMEPAGE = ""

_LICENSE = ""

# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "metaphor_dataset": "",
}


class TRBLLDataset(datasets.GeneratorBasedBuilder):
    """This is a dataset of songs from the genius website"""
    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="TRBLLDataset", version=VERSION, description="songs with annotations dataset"),
    ]

    DEFAULT_CONFIG_NAME = "TRBLLDataset"

    # def _info(self):
    #     features = datasets.Features(
    #         {
    #             "data": datasets.Sequence(datasets.Value("string")),
    #             "labels":datasets.Sequence(datasets.Value("string")),
    #         }
    #     )
    #     return datasets.DatasetInfo(
    #         description=_DESCRIPTION,
    #         features=features,
    #     )

    def _split_generators(self, dl_manager):
        with open('config.yaml') as f:
            training_args = yaml.load(f, Loader=yaml.FullLoader)

        data_dir = training_args["train_args"]["data_dir"]
        dataset_dir = training_args["train_args"]["dataset_dir"]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, dataset_dir, "_train.json"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, dataset_dir, "_test.json"),
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, dataset_dir, "_validation.json"),
                    "split": "validation",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        df = pd.read_json(filepath)
        for index, row in df.iterrows():
            yield index, {
                "data": row['data'],
                "labels": row['labels'],
            }
