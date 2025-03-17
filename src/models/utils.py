from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from src.utils.system import VECTOR_HF_MAPPING, Hosts, get_host


def load_tokenizer(model_name: str) -> AutoTokenizer:
    kwargs = dict()
    model_path = model_name
    if get_host() == Hosts.vector:
        model_path = VECTOR_HF_MAPPING.get(model_name, model_path)
        kwargs["local_files_only"] = True
    return AutoTokenizer.from_pretrained(model_path, **kwargs)


def load_model(model_name, dtype):
    print("Load modelll")
    # First check if CUDA is available
    if torch.cuda.is_available():
        device_map = "auto"  # Use all available GPUs
    else:
        device_map = "cpu"  # Force CPU if no GPU available

    kwargs = dict(device_map=device_map)
    # kwargs = dict(device_map="cpu")

    model_path = model_name
    if get_host() == Hosts.vector:
        model_path = VECTOR_HF_MAPPING.get(model_name, model_path)
        kwargs["local_files_only"] = True

    if dtype == "bfloat16":
        kwargs["torch_dtype"] = torch.bfloat16

    # Remove the incorrect map_location parameter
    # kwargs["map_location"] = lambda storage, loc: storage.cuda() if torch.cuda.is_available() else storage.cpu()

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    return model


def load_vllm_model(
    model_name,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    quantization: Optional[str] = None,  # Options: "awq", "sq", None
    dtype: Optional[Literal["bfloat16", "float16", "float32"]] = "bfloat16",
):
    kwargs = dict(trust_remote_code=True)
    model_path = model_name
    if get_host() == Hosts.vector:
        model_path = VECTOR_HF_MAPPING.get(model_name, model_path)
        kwargs["trust_remote_code"] = True
        # kwargs["load_format"] = "hf"

    if dtype == "bfloat16":
        kwargs["dtype"] = "bfloat16"

    model = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        quantization=quantization,
        **kwargs,
    )
    return model
