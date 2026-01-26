#!/bin/bash
set -x
cd ${AGENT_DIR}
eval "$(conda shell.bash hook)"
conda activate agent

python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'WARNING: No GPU')"
python -c "import tensorflow as tf; print('GPUs Available: ', tf.config.list_physical_devices('GPU'))"

# these need to pre-exist for the symbolic links to work
mkdir -p ${AGENT_DIR}/working/bench/

ln -sf /home/logs ${AGENT_DIR}/working/bench/logs
ln -sf /home/submission ${AGENT_DIR}/working/bench/submission

# run with timeout, and print if timeout occurs
timeout $TIME_LIMIT_SECS uv run python -m src.main $@
if [ $? -eq 124 ]; then
  echo "Timed out after $TIME_LIMIT"
fi
