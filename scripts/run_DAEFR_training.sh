#export BASICSR_JIT=True
export CXX=g++

# Increase NCCL timeout to prevent timeout during torch.compile
export NCCL_TIMEOUT=3600  # 1 hour instead of default 30 minutes
export TORCH_NCCL_BLOCKING_WAIT=0  # Non-blocking to prevent deadlock
export NCCL_ASYNC_ERROR_HANDLING=1  # Async error handling to avoid blocking

# Memory optimization: reduce fragmentation for large allocations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

conf_name='DAEFR'

ROOT_PATH='./experiments/' # The path for saving model and logs

# gpus='0,1,2,3'  # Multi-GPU (original)
gpus='0'  # Single B200 with 192GB VRAM - batch_size=32, bf16-mixed

#P: pretrain SL: soft learning
node_n=1

python -u main_DAEFR.py \
--root-path $ROOT_PATH \
--base 'configs/'$conf_name'.yaml' \
-t True \
--gpus $gpus \
--num-nodes $node_n \
