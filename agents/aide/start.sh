#!/bin/bash
set -x
cd ${AGENT_DIR}
eval "$(conda shell.bash hook)"
conda activate agent

python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'WARNING: No GPU')"
python -c "import tensorflow as tf; print('GPUs Available: ', tf.config.list_physical_devices('GPU'))"

# these need to pre-exist for the symbolic links to work
mkdir -p ${AGENT_DIR}/workspaces/bench
mkdir -p ${AGENT_DIR}/logs

ln -sf /home/logs ${AGENT_DIR}/logs/bench
ln -sf /home/submission/submission.py ${AGENT_DIR}/logs/bench/best_solution.py
# run with timeout, and print if timeout occurs
timeout $TIME_LIMIT_SECS aide $@
if [ $? -eq 124 ]; then
  echo "Timed out after $TIME_LIMIT"
fi
