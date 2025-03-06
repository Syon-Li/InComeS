# Your path: /apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/, /zzz/zhisonzhang/users/shuaiyili/

# model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# model_path = "TinyLlama/TinyLlama_v1.1"
# model_path = "meta-llama/Llama-3.2-1B"
# model_path = "meta-llama/Llama-2-7b-hf"
# model_path = "meta-llama/Llama-2-7b-chat-hf"
# model_path = "EleutherAI/gpt-j-6B"


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


# echo clear 0 1 2 3 4 5 6 7 >~/.cfile
# python -m accelerate.commands.launch --config_file ./deepspeed_config.yaml ./edit_train_query.py --model Qwen/Qwen2.5-7B --gradaccu 1 --gist_bsz 256 --gist_num 0 > ./logs/log_$(date +%Y-%m-%d_%H-%M-%S) 2>&1
# echo >~/.cfile


echo clear 0 1 2 3 4 5 6 7 >~/.cfile
python -m accelerate.commands.launch --config_file ./deepspeed_config.yaml ./edit_train_query_fused.py --model Qwen/Qwen2.5-7B --gradaccu 1 --gist_bsz 256 --gist_num 0 > ./logs/log_$(date +%Y-%m-%d_%H-%M-%S) 2>&1
echo >~/.cfile












# python /apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/forward_check.py > logs/log_$(date +%Y-%m-%d_%H-%M-%S) 2>&1


# python data_prep.py --proportion 0.016 --input_ffile /apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/slimpajama_train_file.json --split 64 --split_i 4

# python --mode ffile

# for i in {0..199}; do
# python data_prep.py --mode tokenize --proportion 0.016 --input_ffile /apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/slimpajama_train_file.json --split 200 --split_i ${i} > log.s${i} &
# done


# for ii in 0 1 2 3 4 5 6 7; do echo clear ${ii} >/jizhi/jizhi2/worker/trainer/.cfile${ii}; done
# python -m accelerate.commands.launch --config_file ./fsdp_config.yaml ./edit_train_query.py --model EleutherAI/gpt-j-6B --gradaccu 1 --gist_bsz 256 --txt_csz 0 > ./logs/log_$(date +%Y-%m-%d_%H-%M-%S) 2>&1
# for ii in 0 1 2 3 4 5 6 7; do echo >/jizhi/jizhi2/worker/trainer/.cfile${ii}; done
