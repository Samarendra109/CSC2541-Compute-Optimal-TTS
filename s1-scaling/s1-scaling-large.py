from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_from_disk

import json
from tqdm import tqdm
import re

import sys



# Decide on a token limit for thinking; As the model's max tokens is 32768, 32000 usually ensures there is enough space for the model to still answer
MAX_TOKENS_THINKING = 32000
# Decide how often to ignore end-of-thinking token
NUM_IGNORE = int(sys.argv[1])

early_stops = 0

ds = load_from_disk('~/scratch/datasets/GSM8K')['test']



model = LLM(
    model = "/home/a254liu/scratch/models/s1-32B", # s1 originally gets this prompt wrong but with budget forcing it fixes it
    tensor_parallel_size=4
)
tok = AutoTokenizer.from_pretrained(
    "simplescaling/s1-32B"
)

stop_token_ids = tok("<|im_end|>")["input_ids"]
sampling_params = SamplingParams(
    max_tokens=32768,
    min_tokens=0,
    stop_token_ids=stop_token_ids,
    skip_special_tokens=False,
    temperature=0.0,
)

# For the exact raspberry sample in the paper see
prompts = ds['question']
true_answers_orig = ds['answer']
true_answers = []
for elem in true_answers_orig:
    index = elem.rfind('###')
    true_answers.append(elem[index + 3:].strip())


answers = []

for i, p in enumerate(tqdm(prompts)):
    prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant. Place ### before the final answer to indicate it. <|im_end|>\n<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n"
    stop_token_ids = tok("<|im_start|><|im_end|>")["input_ids"]
    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS_THINKING,
        min_tokens=0,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=False,
        temperature=0.0,
    )
    prompt += "<|im_start|>think"
    o = model.generate(
        prompt,
        sampling_params=sampling_params
    )
    ignore_str = "Wait"
    max_tokens_thinking_tmp = MAX_TOKENS_THINKING
    if max_tokens_thinking_tmp > 0:
        for i in range(NUM_IGNORE): # Num of times to skip stop token
            max_tokens_thinking_tmp -= len(o[0].outputs[0].token_ids)
            if max_tokens_thinking_tmp <= 0:
                early_stops += 1
                break
            prompt += o[0].outputs[0].text + ignore_str
            sampling_params = SamplingParams(
                max_tokens=max_tokens_thinking_tmp,
                min_tokens=1,
                stop_token_ids=stop_token_ids,
                skip_special_tokens=False,
                temperature=0.0,
            )
            o = model.generate(
                prompt,
                sampling_params=sampling_params
            )
            
    ### Final answer ###
    prompt += o[0].outputs[0].text # You can also append "Final Answer:" here like we do for some evaluations to prevent the model from just continuing to reason in its answer when early exiting
    stop_token_ids = tok("<|im_end|>")["input_ids"]
    sampling_params = SamplingParams(
        max_tokens=32768,
        min_tokens=0,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=False,
        temperature=0.0,
    )
    o = model.generate(
        prompt,
        sampling_params=sampling_params,
    )
    #print("With budget forcing:") # You will see that after the "Wait" in the reasoning trace it fixes its answer
    #print(prompt + o[0].outputs[0].text)
    final_output = prompt + o[0].outputs[0].text
    answer = re.findall(r"[-+]?\d*\.\d+|\d+", final_output)
    if answer:
        answers.append(answer[-1])
    else:
        answers.append("<BAD OUTPUT>")
    

with open('GSM8K_answers.json', 'w') as f:
    json.dump(answers, f)

assert(len(answers) == len(true_answers))


correct = 0
total = 0
for i in range(len(answers)):
    if answers[i] == '<BAD OUTPUT>':
        continue
    total += 1
    if answers[i] == true_answers[i]:
        correct += 1

print(correct / total)
print(total)
print(early_stops)


