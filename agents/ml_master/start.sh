#!/bin/bash
set -x # Print commands and their arguments as they are executed

cd ${AGENT_DIR}
eval "$(conda shell.bash hook)"
conda activate agent
MEMORY_INDEX=0
start_cpu=0
CPUS_PER_TASK=32
end_cpu=$((start_cpu + CPUS_PER_TASK - 1))

export MEMORY_INDEX
format_time() {
  local time_in_sec=$1
  local hours=$((time_in_sec / 3600))
  local minutes=$(((time_in_sec % 3600) / 60))
  local seconds=$((time_in_sec % 60))
  echo "${hours}hrs ${minutes}mins ${seconds}secs"
}
export TIME_LIMIT=$(format_time ${TIME_LIMIT_SECS})

mkdir -p ${AGENT_DIR}/logs/run
mkdir -p ${AGENT_DIR}/workspaces/run/bench/best_solution

ln -sf /home/logs ${AGENT_DIR}/logs/run/bench
touch /home/submission/submission.py
ln -sf /home/submission/submission.py ${AGENT_DIR}/workspaces/run/bench/best_solution/solution.py

CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} timeout $TIME_LIMIT_SECS python main_mcts.py \
  start_cpu_id="${start_cpu}" \
  cpu_number="${CPUS_PER_TASK}" \
  agent.code.base_url=${CODE_BASE_URL} \
  agent.code.api_key=${CODE_API_KEY} \
  agent.feedback.base_url=${FEEDBACK_BASE_URL} \
  agent.feedback.api_key=${FEEDBACK_API_KEY} \
  $@

if [ $? -eq 124 ]; then
  echo "Timed out after $TIME_LIMIT"
fi