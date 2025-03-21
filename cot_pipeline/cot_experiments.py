from fewshot_pipeline import get_fewshot_prompt_dataset, get_fewshot_prompt_query_target
from datasets import load_dataset
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def preprocess_gsm8k(examples):

    answer: str = examples['answer']
    reasoning, answer = answer.split("####")

    return {
        "reasoning": reasoning.strip(),
        "target": answer.strip()
    }


if __name__ == '__main__':

    ds_fewshot = load_dataset("openai/gsm8k", 'main', split='train').map(preprocess_gsm8k)
    ds_main = load_dataset("openai/gsm8k", 'main', split='test').map(preprocess_gsm8k)

    prompt_dataset = get_fewshot_prompt_dataset(
        ds_main, n=3, fewshot_dataset=ds_fewshot
    ).map(partial(get_fewshot_prompt_query_target, n=3, query_col='question', answer_col='target', cot_col='reasoning'))

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    instructions_to_model = " ".join((
        "You are provided with a few math questions. The question is given following the Query: tag.",
        "The reaosning is given following Reasoning: tag. And the answer is given following Answer: tag.",
        "Answer the final question provided to you. Provide your reasoning following Reasoning: tag.",
        "Then give a single number as the answer following Answer: tag."
    ))

    for prompt in tqdm(prompt_dataset['prompt']):
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. "+instructions_to_model},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        pass

