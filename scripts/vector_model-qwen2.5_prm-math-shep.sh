#!/bin/bash
#SBATCH --job-name=openr
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-gpu=2
##SBATCH --gres=gpu:a40:2
##SBATCH --cpus-per-task=2
##SBATCH --qos=m2
#SBATCH --exclude=gpu038

### Make sure to define BASE_DIR, VALUE_MODEL_NAME, POLICY_MODEL_NAME, METHOD

# Function to check if models are registered with controller
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR="/h/altintas/llm-tts/scripts/"
echo $SCRIPT_DIR


set -e
METHOD="bon"
METHOD="beam_search"

NUM_PARAMS=${1-32B}
METHOD=${2-bon}
BEAM_WIDTH=${3-"none"}
BASE_DIR="$HOME/llm-tts"
cd $BASE_DIR
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/openr

cd openr

HOST_ADDR=0.0.0.0
CONTROLER_PORT=28777
WORKER_BASE_PORT=30010
WORKER_BASE_PORT=$((SLURM_JOB_ID % 65000))

echo PYTHON_EXECUTABLE=$(which python3)
PYTHON_EXECUTABLE=$(which python3)

CUDA_DEVICE_BASE=0

VALUE_MODEL_NAME="math-shepherd-mistral-7b-prm"
VALUE_MODEL_PATH="$SCRATCH/$VALUE_MODEL_NAME"
# VALUE_MODEL_NAME="Qwen2.5-Math-PRM-7B"
# VALUE_MODEL_PATH="/model-weights/$VALUE_MODEL_NAME"
POLICY_MODEL_NAME="Qwen2.5-0.5B-Instruct"
# POLICY_MODEL_NAME="Qwen2.5-1.5B-Instruct"
POLICY_MODEL_NAME="Qwen2.5-3B-Instruct"
POLICY_MODEL_NAME="Qwen2.5-7B-Instruct"
POLICY_MODEL_NAME="Qwen2.5-14B-Instruct"
POLICY_MODEL_NAME="Qwen2.5-32B-Instruct"
POLICY_MODEL_NAME="Qwen2.5-32B-Instruct"
POLICY_MODEL_NAME="Qwen2.5-${NUM_PARAMS}-Instruct"
MODEL_PATH="/model-weights/$POLICY_MODEL_NAME"

# 0.5B -> 15309271 [DONE]
# 7B -> 15310174 [RUNNING]
# 32B -> 15310262 , a100: 15310505

### beam_search
# 0.5B -> 15326681 [DONE]
# 7B -> 15326678 [DONE]p
# 32B -> 


## modified
### beam_search
# 0.5 -> 15624928
# 1.5 -> 15624926
# 3 -> 15624929
# 7 -> 15624931
# 14 -> 15624934
# 32 -> 15624936

check_models_ready() {
    $PYTHON_EXECUTABLE "$SCRIPT_DIR/check_models_ready.py" --host $HOST_ADDR --controller_port $CONTROLER_PORT
}

LOGDIR=logs_fastchat

tmux start-server
tmux new-session -s FastChat1-$SLURM_JOB_ID -n controller -d
tmux send-keys "export LOGDIR=${LOGDIR}" Enter
tmux send-keys "$PYTHON_EXECUTABLE -m fastchat.serve.controller --port ${CONTROLER_PORT} --host $HOST_ADDR" Enter

NUM_LM_WORKER=${4-1}
NUM_RM_WORKER=1
RESUME_DIR=${5-""}
RESUME_STR=""

if [[ $RESUME_DIR != "" ]]; then
    echo Resuming from $RESUME_DIR
    RESUME_STR="  --resume_dir $RESUME_DIR  "
fi

echo "Wait 10 seconds ..."
sleep 5


num_gpus=1

if [[ "$BEAM_WIDTH" != "none" ]]; then
    beam_width=$BEAM_WIDTH
elif [[ "$POLICY_MODEL_NAME" =~ "0.5B" ]] ;then 
    beam_width=64
elif [[ "$POLICY_MODEL_NAME" =~ "1.5B" ]] ;then 
    beam_width=32
elif [[ "$POLICY_MODEL_NAME" =~ "3B" ]] ;then 
    beam_width=16
elif [[ "$POLICY_MODEL_NAME" =~ "7B" ]] ;then 
    beam_width=8
elif [[ "$POLICY_MODEL_NAME" =~ "14B" ]] ;then 
    beam_width=4
elif [[ "$POLICY_MODEL_NAME" =~ "30B" ]] || [[ "$POLICY_MODEL_NAME" =~ "32B" ]]; then
    beam_width=2
elif [[ "$POLICY_MODEL_NAME" =~ "70B" ]] || [[ "$POLICY_MODEL_NAME" =~ "72B" ]]; then
    beam_width=1
fi

if [[ "$POLICY_MODEL_NAME" =~ "30B" ]] || [[ "$POLICY_MODEL_NAME" =~ "32B" ]]; then
    num_gpus=2
elif [[ "$POLICY_MODEL_NAME" =~ "70B" ]] || [[ "$POLICY_MODEL_NAME" =~ "72B" ]]; then
    num_gpus=2
fi
echo Running with beam width $beam_width

echo "Starting workers"
echo Distributing policy model on $num_gpus gpus.

for i in $(seq 0 $((NUM_LM_WORKER-1)))
do
    # Create proper GPU list based on num_gpus
    GPU_LIST=""
    for j in $(seq 0 $(($num_gpus-1))); do
        if [ $j -eq 0 ]; then
            GPU_LIST="$((i+j))"
        else
            GPU_LIST="$GPU_LIST,$((i+j))"
        fi
    done
  WORKER_PORT=$((WORKER_BASE_PORT+i))
  tmux new-window -n policy_worker_$i
  tmux send-keys "export LOGDIR=${LOGDIR}" Enter
  tmux send-keys "VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=$GPU_LIST $PYTHON_EXECUTABLE -m reason.llm_service.workers.vllm_worker --model-path $MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT --dtype bfloat16 --num-gpus=$num_gpus --swap-space 4 " Enter
done

# start value service
for i in $(seq 0 $((NUM_RM_WORKER-1)))
do
    WORKER_PORT=$((i+WORKER_BASE_PORT+num_gpus*NUM_LM_WORKER))
  tmux new-window -n value_worker
  tmux send-keys "export LOGDIR=${LOGDIR}" Enter
  tmux send-keys "CUDA_VISIBLE_DEVICES=$((i+NUM_LM_WORKER*num_gpus+CUDA_DEVICE_BASE)) $PYTHON_EXECUTABLE -m reason.llm_service.workers.reward_model_worker --model-path $VALUE_MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT  " Enter
done




bon() {
    python reason/evaluation/evaluate.py \
    --LM $POLICY_MODEL_NAME \
    --RM $VALUE_MODEL_NAME \
    --task_name MATH \
    --temperature 0.7 \
    --num_sequence $beam_width \
    --max_new_tokens 2048 \
    --save_dir debug \
    --method best_of_n \
    --num_worker 4  $RESUME_STR \
    --controller_addr http://0.0.0.0:28777 
}

beam_search() {
    python reason/evaluation/evaluate.py \
    --LM $POLICY_MODEL_NAME \
    --RM $VALUE_MODEL_NAME \
    --task_name MATH \
    --temperature 0.7 \
    --max_new_tokens 2048 \
    --num_sequence 1 \
    --tree_max_width $beam_width \
    --tree_max_depth 50 \
    --save_dir debug \
    --method beam_search \
    --num_worker 4 $RESUME_STR \
    --controller_addr http://0.0.0.0:28777
}

echo "Waiting for all models to initialize..."
check_models_ready

if [ $? -ne 0 ]; then
  echo "Failed to initialize all models"
  exit 1
fi

echo "All models initialized. Running main evaluation script..."

if [[ "$METHOD" == "bon" ]]; then
    bon
elif [[ "$METHOD" == "beam_search" ]]; then
    beam_search
fi

exit 0