import torch
from datasets import Dataset, load_dataset, concatenate_datasets
import math
from functools import partial

def get_fewshot_prompt_dataset(
        dataset: Dataset,
        n: int,
        fewshot_dataset: Dataset = None, 
) -> Dataset:
    """
    Constructs a few-shot prompt dataset by selecting `n` examples from either the provided `fewshot_dataset`
    or randomly sampled from `dataset` if `fewshot_dataset` is not provided.
    
    Args:
        dataset (Dataset): The main dataset containing query-answer pairs.
        n (int): The number of few-shot examples to include.
        query_col (str): The column name containing query inputs.
        answer_col (str): The column name containing expected answers.
        cot_col (str, optional): The column name containing chain-of-thought reasoning. Defaults to None.
        fewshot_dataset (Dataset, optional): A separate dataset for few-shot examples. Defaults to None.

    Returns:
        Dataset: A concatenated dataset with few-shot examples prepended to each example.
    
    Raises:
        ValueError: If `n` is greater than the size of `dataset` or `fewshot_dataset`.
    """
    if fewshot_dataset is None:
        if n >= len(dataset):
            raise ValueError(
                f"Not enough datapoints for a {n}-shot prompt. "
                f"A dataset of size {len(dataset)} was passed."
            )
        
        index = torch.arange(0, len(dataset))
        n_add_rand = (torch.randperm(len(dataset)-1) + 1)[:n]
        fewshot_dataset = dataset
        
    else:
        if n > len(fewshot_dataset):
            raise ValueError(
                f"Not enough datapoints for a {n}-shot prompt. "
                f"A dataset of size {len(fewshot_dataset)} was passed."
            )
        
        tiling_factor = math.ceil(len(dataset) / len(fewshot_dataset))
        index = torch.arange(0, len(fewshot_dataset)).tile((tiling_factor,))[:len(dataset)]
        n_add_rand = torch.randperm(len(fewshot_dataset))[:n]
        
    fewshot_indices = (index[None, :] + n_add_rand[:, None]) % len(fewshot_dataset)
    fewshot_prompt_dataset_list = []
    for i in range(n):
        ithshot_dataset = fewshot_dataset.select(fewshot_indices[i])
        for col in ithshot_dataset.features:
            ithshot_dataset = ithshot_dataset.rename_column(col, f"{col}_{i}")
        fewshot_prompt_dataset_list.append(ithshot_dataset)

    fewshot_prompt_dataset_list.append(dataset)
    return concatenate_datasets(fewshot_prompt_dataset_list, axis=1)

def get_fewshot_prompt_query_target(
        examples, 
        n: int, 
        query_col: str,
        answer_col: str,
        cot_col: str = None,
):
    """
    Generates a few-shot learning prompt by formatting query-answer pairs from the dataset.
    
    Args:
        examples (dict): A dictionary containing the dataset examples.
        n (int): Number of few-shot examples to include.
        query_col (str): Column name for query inputs.
        answer_col (str): Column name for expected answers.
        cot_col (str, optional): Column name for chain-of-thought reasoning. Defaults to None.
    
    Returns:
        dict: A dictionary containing the formatted few-shot prompt string.
    
    Note:
        To be used with dataset.map function
        Currently, this function cannot be used with `batched=True` in a Hugging Face dataset.
    """
    prompt_str = ""
    for i in range(n):
        query_i = examples[f"{query_col}_{i}"]
        answer_i = examples[f"{answer_col}_{i}"]
        cot_i = examples[f"{cot_col}_{i}"] if cot_col else ""
        
        if cot_i == "":
            prompt_str += "\n".join((
                "Query:",
                query_i,
                "Answer:",
                answer_i,
                ""
            ))
        else:
            prompt_str += "\n".join((
                "Query:",
                query_i,
                "Reasoning:",
                cot_i,
                "Answer:",
                answer_i,
                ""
            ))

    query = examples[query_col]
    prompt_str += "\n".join((
            "Query:",
            query,
        ))
    
    return {"prompt": prompt_str}

if __name__ == '__main__':
    ds = load_dataset("kaist-ai/CoT-Collection", split='train')
    ds_fewshot = ds.filter(lambda example: example['task'] == 'squad_v1')
    ds = ds.filter(lambda example: example['task'] == 'quac')
    
    fs_prompt_dataset = get_fewshot_prompt_dataset(
        ds, n=3, fewshot_dataset=ds_fewshot
    ).map(partial(get_fewshot_prompt_query_target, n=3, query_col='source', answer_col='target', cot_col='rationale'))
    pass
