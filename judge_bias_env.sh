module load miniconda3
module load cuda/12.4
eval "$(conda shell.bash hook)"
conda activate judge-bias

export PYTHONNOUSERSITE=1
export REPO_ROOT=/mnt/beegfsstudents/home/3202029/llm-judge-self-preference
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"

export HF_HOME=/mnt/beegfsstudents/home/3202029/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME=/mnt/beegfsstudents/home/3202029/torch_cache
export VLLM_CACHE_ROOT=/mnt/beegfsstudents/home/3202029/vllm_cache
export PIP_CACHE_DIR=/mnt/beegfsstudents/home/3202029/pip_cache
export HF_HUB_DISABLE_XET=1

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
         "$TORCH_HOME" "$VLLM_CACHE_ROOT" "$PIP_CACHE_DIR"

cd "$REPO_ROOT"
