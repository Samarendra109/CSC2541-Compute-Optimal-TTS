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
python main.py --model_name "Qwen/Qwen2.5-0.5B-Instruct"   --tts_strategy bon   --beam_width 5   --task_name MATH   --use_wandb=false   --save_dir "./results" --verifier_model "Qwen/Qwen2.5-0.5B-Instruct"  
--vllm
```


TODOs
- [ ] Serve models in the cluster