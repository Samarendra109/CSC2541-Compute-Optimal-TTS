from fewshot_pipeline import get_fewshot_prompt_dataset, get_fewshot_prompt_query_target
from datasets import load_dataset
from functools import partial
from transformers import AutoTokenizer
from tqdm import tqdm
import ollama
import json
import sys


def preprocess_gsm8k(examples):

    answer: str = examples['answer']
    reasoning, answer = answer.split("####")

    return {
        "reasoning": reasoning.strip(),
        "target": answer.strip()
    }


def get_tokens_used(prompt, response, tokenizer) -> int:

    messages = prompt + [{"role": response['role'], "content": response['content']}]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return len(tokenizer.tokenize(text))


if __name__ == '__main__':

    model_size = 3
    num_shot = int(sys.argv[1])
    ds_fewshot = load_dataset("openai/gsm8k", 'main', split='train').map(preprocess_gsm8k)
    ds_main = load_dataset("openai/gsm8k", 'main', split='test').map(preprocess_gsm8k)

    prompt_dataset = get_fewshot_prompt_dataset(
        ds_main, n=num_shot, fewshot_dataset=ds_fewshot
    ).map(partial(get_fewshot_prompt_query_target, n=num_shot, query_col='question', answer_col='target', cot_col='reasoning'))

    model_name = f'qwen2.5:{model_size}b'
    model_hf_name = f"Qwen/Qwen2.5-{model_size}B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_hf_name)

    instructions_to_model = " ".join((
        "You are provided with a few math questions. The question is given following the Query: tag.",
        "The reaosning is given following Reasoning: tag. And the answer is given following Answer: tag.",
        "Answer the final question provided to you. Provide your reasoning following Reasoning: tag.",
        "Then give a single number as the answer following Answer: tag."
    ))

    results = []
    ollama.pull(model_name)

    for data_point in tqdm(prompt_dataset):
        prompt = data_point['prompt']
        answer = data_point['target'].strip()
        prompt_messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. "+instructions_to_model},
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        response = ollama.chat(
            model=model_name, 
            messages=prompt_messages, 
            options={
                "seed": 10,
                "temperature": 0.0,
                "top_k":1,
                "num_ctx":128_000,
                "num_predict": 512,
            }
        )['message']

        tokens = get_tokens_used(prompt_messages, response, tokenizer)
        
        model_answer: str = response['content'].split("Answer:")
        if len(model_answer) == 2:
            model_answer = model_answer[-1].strip()
        else:
            model_answer = "None"

        results.append({
            "tokens": tokens,
            "is_correct": answer == model_answer
        })

    with open(f"results/{model_name}_{num_shot}shot.json", "w") as f:
        json.dump(results, f, indent=4)
        

        
        


