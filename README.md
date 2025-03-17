# CSC2541-Compute-Optimal-TTS

Basic python virtual environment (you can alternatively use conda/micromamba etc).

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/openr
uv venv --python 3.10
source .venv/bin/activate
```

## Verifier Models
Install dependencies with uv, make sure you have cuda-12.4
```bash
uv sync --extra build 
uv sync --extra build --extra compile
```

If the above fails, revert to the openr installation.

```bash
python main.py --model_name "Qwen/Qwen2.5-0.5B-Instruct"   --tts_strategy bon   --beam_width 5   --task_name MATH   --use_wandb=false   --save_dir "./results" --verifier_model "Qwen/Qwen2.5-0.5B-Instruct"  --vllm

python main.py --model_name "Qwen/Qwen2.5-0.5B-Instruct"   --tts_strategy bon   --beam_width 5   --task_name MATH   --use_wandb=false   --save_dir "./results" --verifier_model "peiyi9979/math-shepherd-mistral-7b-prm" --vllm --model_gpu_util=0.3 --prm_gpu_util=0.65

```

VLLM version is faster, pass `--vllm` to enable vllm backend.

TODOs
- [X] Serve models in the cluster

Download the mathshepherd model
```bash
huggingface-cli download peiyi9979/math-shepherd-mistral-7b-prm --local-dir $SCRATCH/math-shepherd-mistral-7b-prm --local-dir-use-symlinks False
```

Other useful commands:
```bash
tmux kill-server
tmux attach-session -t FastChat1
bash reason/llm_service/create_service_math_shepherd_dev.sh
```
