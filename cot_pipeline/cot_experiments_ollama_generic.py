from fewshot_pipeline import get_fewshot_prompt_dataset, get_fewshot_prompt_query_target
from datasets import load_dataset
from functools import partial
from transformers import AutoTokenizer
from tqdm import tqdm
import ollama
import json
import sys
import re


QUERY_COL = "problem"
REASONING_COL = "solution"
ANSWER_COL = "answer"
OUTPUT_TOKENS = 2048


def get_tokens_used(prompt, response, tokenizer) -> int:

    messages = prompt + [{"role": response['role'], "content": response['content']}]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return len(tokenizer.tokenize(text))


def get_math500_dataset():
    ds_main = load_dataset("HuggingFaceH4/MATH-500", split='test')
    return ds_main, None, "problem", "solution", "answer", 2048

def get_gsm8k_dataset():

    def preprocess_gsm8k(examples):
        answer: str = examples['answer']
        reasoning, answer = answer.split("####")

        return {
            "reasoning": reasoning.strip(),
            "target": answer.strip()
        }
    
    ds_fewshot = load_dataset("openai/gsm8k", 'main', split='train').map(preprocess_gsm8k)
    ds_main = load_dataset("openai/gsm8k", 'main', split='test').map(preprocess_gsm8k)
    return ds_main, ds_fewshot, "question", "reasoning", "target", 512

def get_dataset_and_col_names(ds_name: str):
    match ds_name:
        case "MATH500":
            return get_math500_dataset()
        case "GSM8K":
            return get_gsm8k_dataset()
    return get_math500_dataset()

def process_output(ds_name, response):

    def get_default_ans():
        return "None", "None"
    
    response = response['content']

    if "Query:" in response:
        response = response.split("Query:")[0]
    
    try:
        if "Answer:" in response:
            if "Reasoning:" in response:
                reasoning_with_answer: str = response.split("Reasoning:", maxsplit=1)[-1]
            else:
                reasoning_with_answer: str = response
            model_reasoning, model_answer = reasoning_with_answer.rsplit("Answer:", maxsplit=1)
            model_reasoning = model_reasoning.strip()
            model_answer = model_answer.strip()
            return model_reasoning, model_answer
        else:
            match ds_name:
                case "MATH500":
                    ans = response.split("boxed")[-1]
                    if not ans:
                        return get_default_ans()
                    if ans[0] == "{":
                        stack = 1
                        a = ""
                        for c in ans[1:]:
                            if c == "{":
                                stack += 1
                                a += c
                            elif c == "}":
                                stack -= 1
                                if stack == 0:
                                    break
                                a += c
                            else:
                                a += c
                    else:
                        return get_default_ans()
                    return response, a
                case "GSM8K":
                    model_reasoning = response
                    model_answer = re.findall(r"[-+]?\d*\.\d+|\d+", response)
                    if model_answer:
                        return model_reasoning, model_answer[-1]
                    else:
                        return get_default_ans()
                case _:
                    return get_default_ans()
    except:
        return get_default_ans()
    

def run_experiments(model_size, num_shot, dataset_name):

    ds_main, ds_fewshot, QUERY_COL, REASONING_COL, ANSWER_COL, OUTPUT_TOKENS = get_dataset_and_col_names(dataset_name)

    prompt_dataset = get_fewshot_prompt_dataset(
        ds_main, n=num_shot, fewshot_dataset=ds_fewshot
    ).map(partial(get_fewshot_prompt_query_target, n=num_shot, query_col=QUERY_COL, answer_col=ANSWER_COL, cot_col=REASONING_COL))

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
        answer = data_point[ANSWER_COL].strip()
        prompt_messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. "+instructions_to_model},
            {"role": "user", "content": prompt}
        ]
        
        response = ollama.chat(
            model=model_name, 
            messages=prompt_messages, 
            options={
                "seed": 10,
                "temperature": 0.0,
                "top_k":1,
                "num_ctx":12_800,
                "num_predict": OUTPUT_TOKENS,
            }
        )['message']

        tokens = get_tokens_used(prompt_messages, response, tokenizer)
        model_reasoning, model_answer = process_output(dataset_name, response)

        results.append({
            "tokens": tokens,
            "prompt": data_point['prompt'],
            "actual_reasoning": data_point[REASONING_COL],
            "model_reasoning": model_reasoning,
            "actual_answer": answer,
            "model_answer": model_answer,
            "model_entire_answer": response['content']
        })

    model_sanitized_name = model_name.replace(":", "_")
    with open(f"results/{model_sanitized_name}_{num_shot}shot_{dataset_name}.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':

    model_size = sys.argv[1]
    num_shots = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #num_shots = [int(sys.argv[2])]
    dataset_names = ["MATH500", "GSM8K"]
    #dataset_names = ['GSM8K']

    for num_shot in num_shots:
        for dataset_name in dataset_names:
            print(f"Running Model_size:{model_size}, Num_shot:{num_shot}, Dataset_name:{dataset_name}")
            run_experiments(model_size, num_shot, dataset_name)
        

        
        


