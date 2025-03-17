import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass
class EvalArgs:
    model_name: str = field(
        default="qwen2.5b", metadata={"help": "Base model used for generation"}
    )
    tts_strategy: Literal["bon", "s1", "cot", "beam", "dvs"] = field(
        default="bon",
        metadata={
            "help": "TTS Strategy, bon: best-of-n, s1, cot: chain of thought, beam: beam search, dvs: diverse verifier tree search"
        },
    )
    tokenizer_name: Optional[str] = field(
        metadata={"help": "Base tokenizer, if not provided inherits from model_name"},
        default=None,
    )
    temperature: float = field(default=0.7, metadata={"help": "Sampling temperature"})
    top_p: float = field(default=0.95, metadata={"help": "Top-p sampling parameter"})
    max_new_tokens: int = field(
        default=8192, metadata={"help": "Maximum number of new tokens to generate"}
    )
    vllm: bool = field(
        default=False, metadata={"help": "Use vLLM for faster inference"}
    )

    verifier_model: str = field(
        default="qwen2.5b", metadata={"help": "Model code to use as the verifier model"}
    )
    beam_width: int = field(
        default=2,
        metadata={
            "help": "Beam width used for beam search or number of samples for best-of-N"
        },
    )
    tree_max_depth: int = field(
        default=1,
        metadata={"help": "Maximum depth of the verifier tree, must be 1 for BoN"},
    )
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    controller_addr: str = "http://0.0.0.0:28778"

    task_name: Literal["gsm8k", "MATH", "aime"] = "gsm8k"
    is_few_shot: bool = field(
        default=False,
        metadata={"help": "Pass true to add few shot examples to the prompt."},
    )

    def __post_init__(self):
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name

        if self.tts_strategy == "bon":
            self.tree_max_depth = 1


@dataclass
class Config:
    use_wandb: bool = False
    wandb_project: str = "tts"
    wandb_entity: str = None
    seed: int = 42
    save_dir: Path = Path("./results")
    resume_dir: Optional[Path] = None

    def __post_init__(self):
        if self.wandb_entity is None:
            self.wandb_entity = os.environ.get("WANDB_ENTITY")
