"""
Wrapper module to create LM and RM calling functions for OpenR.
"""

from typing import Callable, List, Literal, Optional, Tuple, Union

import torch

from openr.reason.evaluation.evaluator import MathEvaluator
from openr.reason.inference.rm_call import DummyRewardModelCaller
from openr.reason.inference.text_generation import ConcatedLMGenResult
from src.models.utils import load_model, load_tokenizer, load_vllm_model

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

import ray

from openr.reason.inference.lm_call import LanguageModelCallingFunction, LMCallingConfig


class LMOutput:
    def __init__(self, text: List[str], num_tokens: int):
        self.text = text
        self.num_tokens = num_tokens

    def first(self) -> Optional[str]:
        return self.text[0] if self.text else None

    def to_dict(self) -> dict:
        return {"text": self.text, "num_tokens": self.num_tokens}


def hf_call(prompt: str, config: LMCallingConfig, model, tokenizer, device):
    ## TODO: (WIP) wrong will need correction and optimize this probably bottleneck
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_tokens = inputs.input_ids.shape[1]

    generation_config = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.temperature > 0,
        "temperature": config.temperature if config.temperature > 0 else 1.0,
        "top_p": config.top_p,
        "top_k": config.top_k if config.top_k else 40,
        "num_return_sequences": config.n,
        "pad_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": True,
        "output_scores": True,  # To get logprobs
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_config)

    # Extract generated sequences
    sequences = outputs.sequences
    scores = outputs.scores if hasattr(outputs, "scores") else None

    # Process each generated sequence
    generated_texts = []
    output_token_lens = []
    cum_logprobs = []

    for i in range(config.n):
        # Get sequence without prompt tokens
        gen_seq = sequences[i, prompt_tokens:]

        # Calculate token length
        output_len = len(gen_seq)
        output_token_lens.append(output_len)

        text = tokenizer.decode(gen_seq, skip_special_tokens=True)
        generated_texts.append(text)

        # Calculate cumulative logprob (if scores available)
        if scores:
            logprobs = []
            for j, logits in enumerate(scores):
                if (
                    i < logits.shape[0]
                ):  # Check if this sequence is still being generated
                    idx = j % config.n  # Map to the correct sequence
                    token_id = sequences[idx, prompt_tokens + j].item()
                    token_logits = logits[idx]
                    token_probs = torch.nn.functional.softmax(token_logits, dim=0)
                    token_logprob = torch.log(token_probs[token_id]).item()
                    logprobs.append(token_logprob)
            cum_logprob = sum(logprobs)
        else:
            cum_logprob = 0.0

        cum_logprobs.append(cum_logprob)

    # Calculate avg logprob by length
    avg_len_logps = [
        clp / max(1, otl) for clp, otl in zip(cum_logprobs, output_token_lens)
    ]

    # Determine finish reasons (simplified)
    finish_reasons = [
        "length" if otl >= config.max_new_tokens else "stop"
        for otl in output_token_lens
    ]

    # Create the ConcatedLMGenResult
    result = ConcatedLMGenResult(
        text=generated_texts,
        prompt_tokens=[prompt_tokens] * config.n,
        num_tokens=output_token_lens,
        cumulative_logprob=cum_logprobs,
        logp_avg_by_len=avg_len_logps,
        finish_reason=finish_reasons,
    )

    return result


def vllm_call(prompt: str, config: LMCallingConfig, llm, tokenizer) -> LMOutput:
    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k if config.top_k else 40,
        max_tokens=config.max_new_tokens,
        n=config.n,
    )
    outputs = llm.generate([prompt] * config.n, sampling_params)
    generated_texts = [output.outputs[0].text for output in outputs]
    token_count = (
        sum(len(tokenizer.encode(text)) for text in generated_texts) // config.n
    )
    return LMOutput(text=generated_texts, num_tokens=token_count)


def create_lm_call(
    model_name: str,
    tokenizer_name: Optional[str] = None,
    use_vllm: bool = False,
    dtype: str = "bfloat16",
) -> Callable:
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(tokenizer_name or model_name)

    if use_vllm and VLLM_AVAILABLE:
        llm = load_vllm_model(model_name, dtype=dtype)
        return lambda prompt, config: vllm_call(prompt, config, llm, tokenizer)
    else:
        model = load_model(model_name, dtype)
        return lambda prompt, config: hf_call(prompt, config, model, tokenizer, device)


def create_rm_call(
    model_name: str,
    tokenizer_name: Optional[str] = None,
    use_vllm: bool = False,
    dtype: str = "bfloat16",
) -> Callable:
    lm_call = create_lm_call(model_name, tokenizer_name, use_vllm, dtype)
    verify_prompt_template = """
Question: {question}

Solution: {solution}

Is the above solution correct? First, verify the reasoning step-by-step. Then, give a score from 0 to 10 where 0 means completely incorrect and 10 means completely correct.

Score:"""

    def rm_call(question: str, partial_sol: str) -> float:
        verify_prompt = verify_prompt_template.format(
            question=question, solution=partial_sol
        )
        config = LMCallingConfig(temperature=0.1, top_p=0.9, max_new_tokens=10, n=1)
        output = lm_call(verify_prompt, config)
        score_text = output.text[0].strip()
        try:
            import re

            score_match = re.search(r"(\d+(\.\d+)?)", score_text)
            if score_match:
                score = float(score_match.group(1))
                score = min(max(score / 10.0, 0.0), 1.0)
            else:
                score = 0.5
        except ValueError:
            score = 0.5
        return score

    return rm_call


class LocalLMCaller(LanguageModelCallingFunction):
    def __init__(
        self,
        model_name: str,
        lm_step_tag: str = None,
        backend: Literal["hf", "vllm"] = "vllm",
    ):
        super().__init__(lm_step_tag)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = load_tokenizer(model_name)
        if backend == "hf":
            self.model = load_model(model_name, dtype="bfloat16")
        elif backend == "vllm":
            self.model = load_vllm_model(model_name, dtype="bfloat16")
        else:
            raise ValueError("Pass hf or vllm as backend.")
        self.backend = backend

    def __call__(self, input_str: str, config: LMCallingConfig) -> LMOutput:
        if self.backend == "vllm":
            return vllm_call(input_str, config, self.model, self.tokenizer)
        return hf_call(input_str, config, self.model, self.tokenizer, self.device)


class LocalRewardModelCaller:
    def __init__(self, config, backend: Literal["hf", "vllm"] = "vllm"):
        self.config = config
        self.step_tag = config.step_tag
        self.format_str = config.format_str
        self.model = LocalLMCaller(
            config.model_name, lm_step_tag=config.step_tag, backend=backend
        )

    def __call__(
        self,
        question_answer_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
        lm_step_tag: str,
    ) -> Union[List[int], List[List[int]]]:
        ##TODO
        if isinstance(question_answer_pairs[0], str):
            response = self.replace_step_tag(question_answer_pairs[1], lm_step_tag)
            input_str = self.format_str.format(
                question=question_answer_pairs[0], answer=response
            )
        else:
            input_str = [
                self.format_str.format(
                    question=s[0],
                    answer=self.replace_step_tag(s[1], lm_step_tag),
                )
                for s in question_answer_pairs
            ]

        output = self.model(input_str, self.config)
        score_text = output.text[0].strip()
        try:
            import re

            score_match = re.search(r"(\d+(\.\d+)?)", score_text)
            if score_match:
                score = float(score_match.group(1))
                score = min(max(score / 10.0, 0.0), 1.0)
            else:
                score = 0.5
        except ValueError:
            score = 0.5
        return score
        if isinstance(question_answer_pairs, tuple):
            question_answer_pairs = [question_answer_pairs]
        scores = []
        for question, answer in question_answer_pairs:
            formatted_answer = self.replace_step_tag(answer, lm_step_tag)
            score = len(formatted_answer) % 10  # Placeholder logic for scoring
            scores.append(score)
        return scores

    def replace_step_tag(self, answer: str, lm_step_tag: str):
        splits = answer.split(lm_step_tag)
        splits = [s.strip() for s in splits]
        response = f" {self.step_tag}".join([s for s in splits if s != ""])
        response += f" {self.step_tag}"
        return response


# @ray.remote
@ray.remote(num_gpus=torch.cuda.device_count())  # This is crucial
class NewRemoteMathEvaluator(MathEvaluator):
    def __init__(
        self,
        task: str,
        lm_call: Union[str, LanguageModelCallingFunction],
        rm_call: Union[str, LanguageModelCallingFunction] = None,
        backend: Literal["vllm", "hf"] = "hf",
        lm_step_tag: Optional[str] = None,
        rm_config=None,
    ):
        lm_call = LocalLMCaller(lm_call, lm_step_tag=lm_step_tag, backend=backend)
        rm_call = LocalRewardModelCaller(rm_config)

        super().__init__(task, lm_call, rm_call)
        super().__init__(task, lm_call, rm_call)
        super().__init__(task, lm_call, rm_call)
        super().__init__(task, lm_call, rm_call)
        super().__init__(task, lm_call, rm_call)
