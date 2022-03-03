from transformers import GPT2LMHeadModel, GPT2Tokenizer,  GPTNeoForCausalLM
import datasets
import torch
import docx
import datetime


def generate_prompts(lyrics, meaning, artist, title, prompt_type):
    if prompt_type == "lyrics_meaning":
        data = "lyrics: {}. meaning:".format(lyrics)
    elif prompt_type == "song_metadata":
        # Load the songs and annotations
        data = 'Explain the next line from the song "{}", written by {}. Explanation:'.format(title, artist)
    elif prompt_type == "question_context":
        data = 'question: what is the meaning of {} in the song "{}", ' \
               'written by {}? context: {}'.format(meaning, title, artist, lyrics)
    else:  # None: no prompt
        data = lyrics
    return data


models_names = ['EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-2.7B'] # 'gpt2-medium'
prompt_types = ['lyrics_meaning', 'song_metadata', 'question_context']
max_input_length = 512
max_target_length = 512
temperature = 0.9
N = 16
num_return_sequences = 8
torch.manual_seed(42)

for model_name in models_names:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if model_name == 'gpt2-medium':
        model = GPT2LMHeadModel.from_pretrained
    else:
        model = GPTNeoForCausalLM.from_pretrained(model_name)
    print("Model: {} loaded".format(model_name))

    # load datasets
    dataset_name = 'TRBLL_dataset.py'
    samples_dataset = datasets.load_dataset(dataset_name)['train']
    print("Loaded {} samples from {}".format(len(samples_dataset), dataset_name))

    # choose random N samples from the dataset
    samples = torch.randint(0, len(samples_dataset['data']) - 1, (N,))

    # create a doc file to write the generated prompts
    doc = docx.Document()
    doc.add_heading('Predicted annotations by different models, prompts and temperature', 0)

    for index in samples:
        lyrics = samples_dataset['data'][index][0]
        meaning = samples_dataset['labels'][index][0]
        artist = samples_dataset['artist'][index][0]
        title = samples_dataset['title'][index][0]

        for prompt_type in prompt_types:
            print("Generating prompts for {}".format(prompt_type))
            txt = generate_prompts(lyrics, meaning, artist, title, prompt_type)
            input_ids = tokenizer(txt, return_tensors="pt").input_ids

            print("Generating {} prompts for {}".format(num_return_sequences, txt))
            gen_tokens = model.generate(
                input_ids,
                do_sample=True,
                temperature=temperature,
                max_length=max_target_length,
                num_return_sequences=num_return_sequences,
                # top_k=50,
                # top_p=0.95,
            )
            gen_text = tokenizer.batch_decode(gen_tokens)[0]
            print("Generated prompt: {}".format(gen_text))
            for i, sample_output in enumerate(gen_tokens):
                gen_text = tokenizer.decode(sample_output)
                # Save to docx file
                para = doc.add_paragraph("Model: {}, prompt: {}, temperature: {}"
                                         .format(model_name, prompt_type, temperature))
                para.add_run("lyrics: {}. meaning: {} \n".format(lyrics, meaning))
                para.add_run("Gerenated text: {} \n".format(gen_text)).bold = True
                print("Generated text: {}".format(gen_text))
    doc.save('predictions_before_training_{}.docx'.format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    print("Predictions saved to file")


# # -----------------------------------------------
# # https://github.com/huggingface/transformers/issues/5942
# class GPT2FinetunedWithNgrams(GPT2LMHeadModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#
#     def forward(
#             self,
#             input_ids=None,
#             past=None,
#             attention_mask=None,
#             token_type_ids=None,
#             position_ids=None,
#             head_mask=None,
#             inputs_embeds=None,
#             labels=None,
#             use_cache=True,
#     ):
#         temperature = 0.85
#         tmp_input_ids = input_ids
#         max_gen_length = 30
#         counter = 0
#         orig_input_str = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
#         strs_to_join = orig_input_str.split()
#         while counter < max_gen_length:
#             transformer_outputs = self.transformer(
#                 tmp_input_ids,
#                 past=past,
#                 attention_mask=attention_mask,
#                 token_type_ids=token_type_ids,
#                 position_ids=position_ids,
#                 head_mask=head_mask,
#                 inputs_embeds=inputs_embeds,
#                 use_cache=use_cache,
#             )
#
#             hidden_states = transformer_outputs[0]
#             lm_logits = self.lm_head(hidden_states) / (temperature)
#             last_token = lm_logits[:, -1]
#             last_token_softmax = torch.softmax(last_token, dim=-1).squeeze()
#
#             next_token = torch.argmax(last_token_softmax).tolist()
#             next_gen_token_str = self.tokenizer.decode(next_token, clean_up_tokenization_spaces=True).strip()
#             strs_to_join.append(next_gen_token_str)
#
#             new_str_input = ' '.join(strs_to_join)
#             tmp_input_ids = self.tokenizer.encode(new_str_input, return_tensors='pt')
#             counter += 1
#         return new_str_input