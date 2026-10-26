#!/usr/bin/bash

set -euo pipefail

umask 007
 
NGPU=${NGPU:-"4"}
PYTHON_BIN=${PYTHON_BIN:-"python"}
MASTER_PORT=${MASTER_PORT:-"29501"}
PORT=${PORT:-"1106"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train"} # MCP is enabled for all training configs
DATASETS=${DATASETS:-"robotwin:1.0"}

: "${MODEL_PATH:?Set MODEL_PATH to the released zero-wam-pretrain model root}"
export MODEL_PATH

## node setting
num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}
datasets=${DATASETS}

## cmd setting
export TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
"${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.train --config-name "${config_name}" --datasets "${datasets}" "$@"
