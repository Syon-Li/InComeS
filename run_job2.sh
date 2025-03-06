export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
export ftp_proxy="http://star-proxy.oa.com:3128"

export TRANSFORMERS_CACHE="/apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/transformers_cache"
export HF_DATASETS_CACHE="/apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/transformers_cache"
export HF_HOME="/apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/transformers_cache"


# echo clear 0 1 2 3 4 5 6 7 >~/.cfile
# python -m accelerate.commands.launch --config_file ./deepspeed_config.yaml ./edit_train_query.py --model meta-llama/Llama-3.2-1B --gradaccu 1 --gist_bsz 256 --gist_num 0 > ./logs/log_$(date +%Y-%m-%d_%H-%M-%S) 2>&1
# echo >~/.cfile


# echo clear 0 1 2 3 4 5 6 7 >~/.cfile
# python -m accelerate.commands.launch --config_file ./fsdp_config.yaml ./edit_train_query.py --model EleutherAI/gpt-j-6B --gradaccu 1 --gist_bsz 256 --gist_num 0 > ./logs/log_$(date +%Y-%m-%d_%H-%M-%S) 2>&1
# echo >~/.cfile

echo clear 0 1 2 3 4 5 6 7 >~/.cfile
python -m accelerate.commands.launch --config_file ./deepspeed_config.yaml ./edit_train_query_fused.py --model Qwen/Qwen2.5-7B --gradaccu 1 --gist_bsz 256 --gist_num 0 > ./logs/log_$(date +%Y-%m-%d_%H-%M-%S) 2>&1
echo >~/.cfile



# echo clear 0 1 2 3 4 5 6 7 >~/.cfile
# python -m accelerate.commands.launch --config_file ./deepspeed_config.yaml ./edit_train_query.py --model EleutherAI/gpt-j-6B --gradaccu 1 --gist_bsz 256 --gist_num 0 > ./logs/log_$(date +%Y-%m-%d_%H-%M-%S) 2>&1
# echo >~/.cfile
