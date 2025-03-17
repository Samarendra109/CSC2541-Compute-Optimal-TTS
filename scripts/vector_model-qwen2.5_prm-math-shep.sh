#!/bin/bash
#SBATCH --job-name=openr
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:a40:2

### Make sure to define BASE_DIR, VALUE_MODEL_NAME, POLICY_MODEL_NAME, METHOD

BASE_DIR="$HOME/llm-tts"
cd $BASE_DIR
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/openr

cd openr

set -e
METHOD="bon"

HOST_ADDR=0.0.0.0
CONTROLER_PORT=28777
WORKER_BASE_PORT=30010

echo PYTHON_EXECUTABLE=$(which python3)
PYTHON_EXECUTABLE=$(which python3)

CUDA_DEVICE_BASE=0

VALUE_MODEL_NAME="math-shepherd-mistral-7b-prm"
POLICY_MODEL_NAME="Qwen2.5-1.5B-Instruct"
MODEL_PATH="/model-weights/$POLICY_MODEL_NAME"
VALUE_MODEL_PATH="$SCRATCH/math-shepherd-mistral-7b-prm"


bon() {
    python reason/evaluation/evaluate.py \
    --LM $POLICY_MODEL_NAME \
    --RM $VALUE_MODEL_NAME \
    --task_name MATH \
    --temperature 0.7 \
    --num_sequence 8 \
    --max_new_tokens 2048 \
    --save_dir debug \
    --method best_of_n \
    --num_worker 2 \
    --controller_addr http://0.0.0.0:28777
}

LOGDIR=logs_fastchat

tmux start-server
tmux new-session -s FastChat1 -n controller -d
tmux send-keys "export LOGDIR=${LOGDIR}" Enter
tmux send-keys "$PYTHON_EXECUTABLE -m fastchat.serve.controller --port ${CONTROLER_PORT} --host $HOST_ADDR" Enter

NUM_LM_WORKER=1
NUM_RM_WORKER=1

echo "Wait 10 seconds ..."
sleep 5

echo "Starting workers"
for i in $(seq 0 $((NUM_LM_WORKER-1)))
do
  WORKER_PORT=$((WORKER_BASE_PORT+i))
  tmux new-window -n policy_worker_$i
  tmux send-keys "export LOGDIR=${LOGDIR}" Enter
  tmux send-keys "CUDA_VISIBLE_DEVICES=$((i+CUDA_DEVICE_BASE)) $PYTHON_EXECUTABLE -m reason.llm_service.workers.vllm_worker --model-path $MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT --dtype bfloat16 --swap-space 32" Enter
done

# start value service
for i in $(seq 0 $((NUM_RM_WORKER-1)))
do
  WORKER_PORT=$((i+WORKER_BASE_PORT+NUM_LM_WORKER))
  tmux new-window -n value_worker
  tmux send-keys "export LOGDIR=${LOGDIR}" Enter
  tmux send-keys "CUDA_VISIBLE_DEVICES=$((i+NUM_LM_WORKER+CUDA_DEVICE_BASE)) $PYTHON_EXECUTABLE -m reason.llm_service.workers.reward_model_worker --model-path $VALUE_MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT" Enter
done


# Function to check if models are registered with controller
check_models_ready() {
  CONTROLLER_URL="http://${HOST_ADDR}:${CONTROLER_PORT}"
  echo "Checking models at: $CONTROLLER_URL"
  
  $PYTHON_EXECUTABLE -c "
import requests
import time
import sys

controller_addr = 'http://0.0.0.0:28777'  # Hardcoded to avoid bash variable issues
target_models = ${NUM_LM_WORKER:-1} + ${NUM_RM_WORKER:-1}

def check_models():
    try:
        print(f'Checking {controller_addr}/list_models')
        response = requests.post(controller_addr + '/list_models')
        if response.status_code == 200:
            models = response.json()['models']
            print(f'Found {len(models)} models: {models}')
            return len(models) >= target_models
        else:
            print(f'Error status: {response.status_code}')
        return False
    except Exception as e:
        print(f'Error checking models: {e}')
        return False

# Check immediately once
check_models()

for i in range(120):
    if check_models():
        print('All models ready!')
        sys.exit(0)
    print(f'Waiting for models... {i}/120')
    time.sleep(10)

sys.exit(1)
"
}
check_models_ready() {
  $PYTHON_EXECUTABLE -c "
import requests
import time
import sys

def check_models(controller_addr='http://${HOST_ADDR}:${CONTROLER_PORT}'):
    print(controller_addr)
    try:
        response = requests.post(controller_addr + '/list_models')
        if response.status_code == 200:
            models = response.json()['models']
            print(models)
            return len(models) >= $((NUM_LM_WORKER + NUM_RM_WORKER))
        return False
    except:
        return False

# Wait up to 20 minutes for models to be ready
for i in range(120):
    if check_models():
        print('All models ready!')
        sys.exit(0)
    print(f'Waiting for models... {i}/120')
    time.sleep(10)

print('Timed out waiting for models to initialize')
sys.exit(1)
"
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
fi

exit 0