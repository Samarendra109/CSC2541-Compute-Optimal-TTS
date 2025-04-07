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



Best of N MATH
- [ ] 0.5B - N=64

- [ ] 0.5B - N=1

- [ ] 1.5B - N=32

- [ ] 1.5B - N=1

- [ ] 3B - N=16

- [ ] 3B - N=1

- [ ] 7B - N=8

- [ ] 7B - N=1

- [ ] 14B - N=4

- [ ] 14B - N=1

- [ ] 32B - N=2

- [ ] 32B - N = 1

Beam search
- [ ] 0.5B - beam width = 64

- [ ] 0.5B - beam width = 2

- [ ] 1.5B - beam width = 32

- [ ] 1.5B - beam width = 2

- [ ] 3B -  beam width = 16

- [ ] 3B - beam width = 2

- [ ] 7B -  beam width = 8

- [ ] 7B - beam width = 2

- [ ] 14B - beam width = 4

- [ ] 14B - beam width = 2

- [ ] 32B - beam width = 2

- [ ] 32B - beam width = 2


submit all
```bash
for method in bon beam_search; do
echo $method
for i in 0.5 1.5 3 7 14 32; do
 jobid=$(sbatch --parsable scripts/vector_model-qwen2.5_prm-math-shep.sh  "${i}B" "$method"  )
echo "- [ ] ${i}B - ${jobid}"
done
done

i=0.5
method="bon"
bash scripts/vector_model-qwen2.5_prm-math-shep.sh  "${i}B" "$method" 1

i=0.5
method="beam_search"
bash scripts/vector_model-qwen2.5_prm-math-shep.sh  "${i}B" "$method"
```
bon
- [ ] 0.5B - 15627526
- [ ] 1.5B - 15627527
- [ ] 3B - 15627528
- [ ] 7B - 15627529
- [ ] 14B - 15627530
- [ ] 32B - 15627531
beam_search
- [ ] 0.5B - 15627532
- [ ] 1.5B - 15627533
- [ ] 3B - 15627534
- [ ] 7B - 15627535
- [ ] 14B - 15627536
- [ ] 32B - 15627537

## Baseline
bon
- [ ] 0.5B - 15626051
- [ ] 1.5B - 15626052
- [ ] 3B - 15626053
- [ ] 7B - 15626054
- [ ] 14B - 15626055
- [ ] 32B - 15626056
beam_search
- [ ] 0.5B - 15626057
- [ ] 1.5B - 15626058
- [ ] 3B - 15626059
- [ ] 7B - 15626060
- [ ] 14B - 15626061
- [ ] 32B - 15626062
