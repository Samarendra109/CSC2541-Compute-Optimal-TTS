import json
import logging
import multiprocessing
import os
import random
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import jsonlines
import numpy as np
import ray
import torch
import torch.multiprocessing as mp
import tree
import wandb
from ray.util.actor_pool import ActorPool
from tqdm import tqdm
from transformers import HfArgumentParser

from openr.reason.evaluation.evaluate import Task
from openr.reason.evaluation.evaluator import MathEvaluator, RemoteMathEvaluator
from openr.reason.evaluation.methods import (
    BasicConfig,
    BeamSearchConfig,
    BestOfNConfig,
    CoTConfig,
    RStarMCTSConfig,
    TreeSearchConfig,
    VanilaMCTSConfig,
    beam_search,
    best_of_n,
    cot,
    rstar_mcts,
    vanila_mcts,
)
from openr.reason.inference.lm_call import (
    LanguageModelCallingFunction,
    LMCallingConfig,
    VLLMRemoteCaller,
)
from openr.reason.inference.rm_call import (
    DummyRewardModelCaller,
    RemoteRewardModelConfig,
    RewardModelCallingFunction,
    RMRemoteCaller,
)
from src.model_wrapper import (
    LocalLMCaller,
    LocalRewardModelCaller,
    NewRemoteMathEvaluator,
    create_lm_call,
    create_rm_call,
)
from src.utils.config import Config, EvalArgs

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


def main():
    mp.set_start_method("spawn")
    # Get the number of GPUs available
    num_gpus = torch.cuda.device_count()
    ray.init(num_gpus=num_gpus)
    # Parse arguments
    parser = HfArgumentParser([Config, EvalArgs])
    config: Config
    eval_args: EvalArgs
    config, eval_args = parser.parse_args_into_dataclasses()

    # Setup random seed
    setup_seed(config.seed)

    # Initialize wandb if enabled
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            config={**asdict(config), **asdict(eval_args)},
            name=f"{eval_args.task_name}-{eval_args.tts_strategy}-{eval_args.model_name.split('/')[-1]}",
        )

    # Create language model calling function
    logger.info(f"Initializing model: {eval_args.model_name}")

    # TODO(ziyu): move into some configuration file
    if "math-shepherd" in eval_args.verifier_model.lower():
        prm_step_tag = "ки\n"
    else:
        # assume qwen
        prm_step_tag = "\n\n\n\n\n "
    prm_format_str = "{question} {answer}"

    if "qwen" in eval_args.model_name.lower():
        lm_step_tag = "\n\n"
    else:
        lm_step_tag = "ки\n"

    ## TODO: find a better way to instantiate the llm_call and rm_call, with HF backend they run okay but
    ## vllm backend needs to be instantiated in the NewRemoteMathEvaluator
    if not eval_args.vllm:
        llm_call = LocalLMCaller(
            eval_args.model_name,
            lm_step_tag=lm_step_tag,
            backend="vllm" if eval_args.vllm else "hf",
            gpu_util=eval_args.model_gpu_util,
        )
    # VLLMRemoteCaller(
    #     eval_args.model_name, eval_args.controller_addr, lm_step_tag=lm_step_tag
    # )
    # llm_call = create_lm_call(
    #     model_name=eval_args.model_name,
    #     tokenizer_name=eval_args.tokenizer_name,
    #     use_vllm=eval_args.vllm,
    #     dtype=eval_args.dtype,
    # )
    # Create reward model calling function if needed
    rm_call = None
    if eval_args.verifier_model:
        logger.info(f"Initializing verifier model: {eval_args.verifier_model}")
        rm_config = RemoteRewardModelConfig(
            step_tag=prm_step_tag,
            format_str=prm_format_str,
            model_name=eval_args.verifier_model,
            controller_addr=eval_args.controller_addr,
        )
        if not eval_args.vllm:
            rm_call = LocalRewardModelCaller(
                rm_config,
                backend="vllm" if eval_args.vllm else "hf",
                gpu_util=eval_args.prm_gpu_util,
            )

    # Configure generation parameters
    gen_config = LMCallingConfig(
        temperature=eval_args.temperature,
        top_p=eval_args.top_p,
        top_k=40,  # Common default if not specified
        max_new_tokens=eval_args.max_new_tokens,
        n=1,  # Will be overridden by specific methods
    )

    # Set up task
    task = Task(task_name=eval_args.task_name, is_few_shot=eval_args.is_few_shot)

    # Set up the method configuration and solver function
    logger.info(f"Using strategy: {eval_args.tts_strategy}")

    method_name = eval_args.tts_strategy
    if method_name == "s1":
        # For single-step, we'll use best_of_n with n=1
        method_config = BestOfNConfig(task_name=eval_args.task_name, num_sequence=1)
        solver_fn = partial(best_of_n, method_config, gen_config)
    elif method_name == "cot":
        method_config = CoTConfig(task_name=eval_args.task_name)
        solver_fn = partial(
            cot, method_config, gen_config, lm_call=llm_call, rm_call=rm_call
        )

    elif method_name == "bon":
        method_config = BestOfNConfig(
            task_name=eval_args.task_name,
            num_sequence=eval_args.beam_width,
        )
        solver_fn = partial(
            best_of_n,
            method_config,
            gen_config,
        )
    elif method_name == "beam":
        method_config = BeamSearchConfig(
            task_name=eval_args.task_name,
            tree_max_depth=eval_args.tree_max_depth,
            tree_max_width=eval_args.beam_width,
            beam_size=eval_args.beam_width,
            init_critic_value=True,
        )
        solver_fn = partial(
            beam_search, method_config, gen_config, lm_call=llm_call, rm_call=rm_call
        )

    elif method_name == "dvs":
        # Using R* MCTS which is similar to diverse verifier search
        method_config = RStarMCTSConfig(
            task_name=eval_args.task_name,
            tree_max_depth=eval_args.tree_max_depth,
            tree_max_width=eval_args.beam_width,
            num_path=eval_args.beam_width,
            init_critic_value=True,
            select_by_prior=False,
        )
        solver_fn = partial(rstar_mcts, method_config, gen_config)

    else:
        raise ValueError(f"Unknown strategy: {method_name}")

    # Set up save directory for results
    save_dir = None
    if config.save_dir:
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(config.save_dir) / task.task_name / method_name / datetime_str
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving the results in {save_dir.as_posix()}")

        # Save configuration
        config_record = {
            "model": eval_args.model_name,
            "verifier_model": eval_args.verifier_model,
            "method": method_name,
            "method_config": asdict(method_config),
            "gen_config": asdict(gen_config),
            "is_few_shot": eval_args.is_few_shot,
            "vllm": eval_args.vllm,
            "dtype": eval_args.dtype,
        }
        json.dump(config_record, open(save_dir / "config.json", "w"))

    # Get test dataset
    test_ds = [p for p in task.test_ds]

    logger.info(f"Evaluating {len(test_ds)} problems")

    # Process problems that were already solved if resuming
    answered_questions = set()
    results = []
    record_writer = None

    if config.resume_dir:
        resume_path = Path(config.resume_dir) / "record.jsonl"
        if resume_path.exists():
            with jsonlines.open(resume_path, "r") as reader:
                for obj in reader:
                    results.append(obj["result"])
                    answered_questions.add(obj["question"])

            logger.info(
                f"Resumed {len(answered_questions)} questions from {config.resume_dir}"
            )
            # Filter out already answered questions
            original_count = len(test_ds)
            test_ds = [p for p in test_ds if p["question"] not in answered_questions]
            logger.info(f"Remaining problems to solve: {len(test_ds)}/{original_count}")

    # Set up the record writer if saving results
    if save_dir:
        record_writer = jsonlines.open(save_dir / "record.jsonl", "w")
        # If resuming, copy existing records to the new file
        if config.resume_dir:
            resume_path = Path(config.resume_dir) / "record.jsonl"
            if resume_path.exists():
                with jsonlines.open(resume_path, "r") as reader:
                    for obj in reader:
                        record_writer.write(obj)
    # Determine the number of workers (processes) -> TODO: I think should be num_gpu in the local setting? but it is based on num_cpus in their serve based setting
    num_workers = os.cpu_count() - 1 if os.cpu_count() > 1 else 1
    num_workers = min(num_workers, 8)  # Limit to a reasonable number
    print(num_workers)
    # num_workers = 1
    num_workers = min(4, num_gpus)

    def parallel_evaluate_dataset(
        solver_fn: Callable,
        task_name: str,
        num_workers: int = 4,
        save_dir: Optional[Path] = None,
        record_writer: Optional[Any] = None,
    ) -> List[Dict]:
        """
        Evaluate problems in parallel using a process pool.
        """
        results = []
        kwargs = dict(
            model_gpu_util=eval_args.model_gpu_util, prm_gpu_util=eval_args.prm_gpu_util
        )
        if eval_args.vllm:
            kwargs["lm_call"] = eval_args.model_name
            kwargs["backend"] = "vllm"
        else:
            kwargs["lm_call"] = llm_call
            kwargs["rm_call"] = rm_call
            kwargs["backend"] = "hf"
        ## TODO: check if it
        actor_pool = ActorPool(
            [
                # MathEvaluator(
                NewRemoteMathEvaluator.remote(
                    task=Task(task_name=task_name, is_few_shot=eval_args.is_few_shot),
                    lm_step_tag=lm_step_tag,
                    rm_config=rm_config,
                    **kwargs,
                )
                for _ in range(num_workers)
            ]
        )
        assert torch.cuda.is_available()
        # Convert test_ds items to a (index, item) tuple
        indexed_test_ds = list(enumerate(test_ds))

        # Use the map_unordered with the index as the key
        res_q = actor_pool.map_unordered(
            lambda p, x: p.evaluate_problem.remote(x[1], solver_fn), indexed_test_ds
        )

        for i, (problem_inst, result, output) in enumerate(
            tqdm(res_q, total=len(test_ds))
        ):
            results.append(result)
            if record_writer:
                obj = {
                    "i": i,
                    "question": problem_inst["question"],
                    "groundtruth": problem_inst["answer"],
                    "result": result,
                    "output": output,
                }
                record_writer.write(obj)
        avg_res = (tree.map_structure(lambda *xs: np.mean(xs), *results),)
        if record_writer:
            json.dump(avg_res, open(save_dir / "avg_result.json", "w"))
        print("Method: {}. Average result: {}".format(method_name, avg_res))
        return results

    # Run parallel evaluation
    if test_ds:
        new_results = parallel_evaluate_dataset(
            solver_fn=solver_fn,
            task_name=eval_args.task_name,
            num_workers=num_workers,
            save_dir=save_dir,
            record_writer=record_writer,
        )
        results.extend(new_results)

    # Close the record writer
    if record_writer:
        record_writer.close()


if __name__ == "__main__":
    main()
