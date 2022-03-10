from transformers import T5TokenizerFast, T5ForConditionalGeneration
from box import Box
import yaml
import datasets
import numpy as np
import random

def prepare_masks(data, labels, min_masks_per_sentence = 1, prob=0.15, max_input_length=512, max_output_length=512):
    # default "id{%d}" - extra_ids default = 100 -> [0, 99]
    mask_template = "id"
    mlm_dict = {}
    data_options = {"data": data, "annotations": labels}
    for data_key, data_opt in data_options.items():
        mlm_data_masked = []
        mlm_labels_masks = []
        for sentence in data_opt:
            # split sentences to list of words
            sentence_split = sentence[0].split()
            # number of masks in each sentence
            num_of_masks_sentence = np.maximum(int(np.floor(len(sentence_split) * prob)), min_masks_per_sentence)
            # random indexes of words in the sentence
            rand_indices_sentence = random.sample(range(0, len(sentence_split)), num_of_masks_sentence)
            rand_indices_sentence.sort()
            # replace words with masks
            new_sentence_words = sentence_split.copy()
            new_sentence_labels = []
            for idx, val in enumerate(rand_indices_sentence):
                mask_id = f"{mask_template}{idx}"
                new_sentence_words[val] = mask_id
                label_mask = mask_id + ' ' + sentence_split[val]
                new_sentence_labels.append(label_mask)

            # create new data and sentence (not lists)
            new_sentence = " ".join(new_sentence_words)
            new_sentence_label = " ".join(new_sentence_labels)
            # save new sentences and labels
            mlm_data_masked.append(new_sentence)
            mlm_labels_masks.append(new_sentence_label)

        # save to data option - dict of dicts
        mlm_dict[f"{data_key}"] = {"masked_sentences": mlm_data_masked,
                                   "masks_labels": mlm_labels_masks}

    return mlm_dict

if __name__ == '__main__':

    with open('config.yaml') as f:
        training_args = Box(yaml.load(f, Loader=yaml.FullLoader))

    model_name = training_args.train_args.model_name
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    # model = T5Model.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # params
    max_input_length = 512
    max_target_length = 512

    # load dataset
    num_samples = 100
    dataset_name = 'TRBLL_dataset_mini.py'
    samples_dataset = datasets.load_dataset(dataset_name)
    data = samples_dataset['test']['data'][:num_samples]
    labels = samples_dataset['test']['labels'][:num_samples]

    # masking with consecutive masks
    mlm_dict = prepare_masks(data, labels)

    # training T5 on MLM task
    # data - songs lyrics, annotations - meanings
    # example: "The <extra_id_0> walks in <extra_id_1> park" -> "<extra_id_0> cute dog <extra_id_1> the <extra_id_2>"
    input_ids = tokenizer([sentence for sentence in mlm_dict['data']['masked_sentences']], return_tensors="pt",
                          max_length=max_input_length, truncation=True,  padding=True).input_ids
    labels_ids = tokenizer([sentence for sentence in mlm_dict['data']['masks_labels']], return_tensors="pt",
                           max_length=max_target_length, truncation=True,  padding=True).input_ids
    outputs = model(input_ids=input_ids, labels=labels_ids)
    loss = outputs.loss
    logits = outputs.logits

    # # inference - input as usual on T5 generate task
    # input_ids = tokenizer(
    #     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
    # ).input_ids  # Batch size 1
    # outputs = model.generate(input_ids)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    # # studies have shown that owning a dog is good for you.