export TRANSFORMERS_CACHE="your/transformers_cache"
export HF_DATASETS_CACHE="your/transformers_cache"
export HF_HOME="your/transformers_cache"

python -m accelerate.commands.launch --config_file ./deepspeed_config.yaml ./edit_train_fused.py --model meta-llama/Llama-3.2-1B --gist_bsz 256 > ./logs/log_$(date +%Y-%m-%d_%H-%M-%S) 2>&1
