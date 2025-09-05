#!/bin/bash
set -x # Print commands and their arguments as they are executed
eval "$(conda shell.bash hook)"
conda activate agent
cp ${AGENT_DIR}/main.py /home/submission/submission.py
python ${AGENT_DIR}/main.py

ls /home/data