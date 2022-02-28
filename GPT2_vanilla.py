from transformers import GPT2LMHeadModel, GPT2Tokenizer # GPTNeoForCausalLM
import datasets
import torch

model_name = "gpt2"  # EleutherAI/gpt-neo-1.3B"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)



num_samples = 100
dataset_name = 'TRBLL_dataset_mini.py'
samples_dataset = datasets.load_dataset(dataset_name)
data = samples_dataset['test']['data'][:num_samples]
labels = samples_dataset['test']['labels'][:num_samples]

max_input_length = 512
max_target_length = 100
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# input_ids = tokenizer([sentence[0] for sentence in data], return_tensors="pt",
#                       max_length=max_input_length, truncation=True, padding=True).input_ids
inputs = tokenizer(data[0][0], return_tensors="pt",
                      max_length=max_input_length,
                      truncation=True)

generation_output = model.generate(**inputs, return_dict_in_generate=True, output_scores=True, max_length=100)
generated_pred = tokenizer.batch_decode(generation_output['sequences'], skip_special_tokens=True)

# outputs = model(**inputs, labels=inputs["input_ids"])
# loss = outputs.loss
# logits = outputs.logits

# gen_tokens = model.generate(
#     input_ids,
#     # do_sample=True,
#     # temperature=0.9,
#     # max_length=max_target_length,
# )
# gen_text = tokenizer.batch_decode(gen_tokens)[0]


# -----------------------------------------------
# https://github.com/huggingface/transformers/issues/5942
class GPT2FinetunedWithNgrams(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def forward(
            self,
            input_ids=None,
            past=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            use_cache=True,
    ):
        temperature = 0.85
        tmp_input_ids = input_ids
        max_gen_length = 30
        counter = 0
        orig_input_str = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        strs_to_join = orig_input_str.split()
        while counter < max_gen_length:
            transformer_outputs = self.transformer(
                tmp_input_ids,
                past=past,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
            )

            hidden_states = transformer_outputs[0]
            lm_logits = self.lm_head(hidden_states) / (temperature)
            last_token = lm_logits[:, -1]
            last_token_softmax = torch.softmax(last_token, dim=-1).squeeze()

            next_token = torch.argmax(last_token_softmax).tolist()
            next_gen_token_str = self.tokenizer.decode(next_token, clean_up_tokenization_spaces=True).strip()
            strs_to_join.append(next_gen_token_str)

            new_str_input = ' '.join(strs_to_join)
            tmp_input_ids = self.tokenizer.encode(new_str_input, return_tensors='pt')
            counter += 1
        return new_str_input