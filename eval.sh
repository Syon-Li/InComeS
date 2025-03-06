export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
export ftp_proxy="http://star-proxy.oa.com:3128"

export TRANSFORMERS_CACHE="/apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/transformers_cache"
export HF_DATASETS_CACHE="/apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/transformers_cache"
export HF_HOME="/apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/transformers_cache"

# ./checkpoints/wikipara-Lamini-nli-MCQ-ZSRE-CF/checkpoints_2025-01-18_10-51/checkpoint_1400/ -1 + deduplicated data + marginal loss
# ./checkpoints/wikipara-Lamini-nli-MCQ-ZSRE-CF/checkpoints_2025-01-15_23-48/checkpoint_1400/ -1 + deduplicated data
# ./checkpoints/wikipara-Lamini-nli-MCQ-ZSRE-CF/checkpoints_2025-01-18_16-22/checkpoint_1400/ -1 + original data
# ./checkpoints/wikipara-Lamini-nli-MCQ-ZSRE-CF/checkpoints_2024-12-28_13-56/checkpoint_3500/ 0 + original data

# ./checkpoints/wikipara-Lamini-nli-MCQ-ZSRE-CF/checkpoints_2025-02-17_00-26/
# ./checkpoints/wikipara-Lamini-nli-MCQ-ZSRE-CF/checkpoints_2025-02-22_16-17/ 32 gist num trained

# portability_inverse_relation, portability_one_hop, portability_subject_replace



# echo clear 0 >~/.cfile
# python ./forward_check.py > ./logs/log_$(date +%Y-%m-%d_%H-%M-%S) 2>&1
# echo >~/.cfile


# eval
export CUDA_VISIBLE_DEVICES=0
echo clear 0 >/jizhi/jizhi2/worker/trainer/.cfile0
python eval.py --method gist --model meta-llama/Llama-3.2-1B --dataset dune_scientific --batch_edit 1 --run_base 0 --bsz 100 --edits 3 --wrt 0 --weight_path ./checkpoints/wikipara-Lamini-nli-MCQ-ZSRE-CF/checkpoints_2025-02-25_22-59/
echo >/jizhi/jizhi2/worker/trainer/.cfile0

# export CUDA_VISIBLE_DEVICES=1
# python eval.py --method gist --model meta-llama/Llama-3.2-1B --dataset counterfact --batch_edit 1 --run_base 0 --bsz 500 --edits 4 --wrt 0 --weight_path ./checkpoints/wikipara-Lamini-nli-MCQ-ZSRE-CF/checkpoints_2025-02-25_12-11/ --checkpoint_tag model


# Baseline
# export CUDA_VISIBLE_DEVICES=0
# echo clear 0 >/jizhi/jizhi2/worker/trainer/.cfile0
# # python Editing_baselines_mquake.py --method GRACE --model meta-llama/Llama-3.2-1B --edits 2 --multiple_edit 0 --bsz 100 --wrt 0
# for ii in {2..4}; do
#     python Editing_baselines_mquake.py --method SERAC --model meta-llama/Llama-3.2-1B --edits $ii --multiple_edit 0 --bsz 100 --wrt 1
# done
# echo >/jizhi/jizhi2/worker/trainer/.cfile0


# export CUDA_VISIBLE_DEVICES=1
# python Editing_baselines_mquake.py --method FT-M --model meta-llama/Llama-3.2-1B --edits 4 --multiple_edit 0 --bsz 100 --wrt 0
# for ii in {2..4}; do
#     python Editing_baselines_mquake.py --method ROME --model EleutherAI/gpt-j-6B --edits $ii --multiple_edit 0 --bsz 100 --wrt 1
# done



# export CUDA_VISIBLE_DEVICES=0
# echo clear 0 >/jizhi/jizhi2/worker/trainer/.cfile0
# python Editing_baselines.py --method SERAC --model meta-llama/Llama-3.2-1B --dataset zsre --batch_edit 0 --sequential_edit 0 --bsz 100 --wrt 1
# # ds_array=("portability_inverse_relation" "portability_one_hop" "portability_subject_replace")
# # for ds_str in "${ds_array[@]}"; do
# #     python Editing_baselines.py --method SERAC --model meta-llama/Llama-3.2-1B --dataset $ds_str --batch_edit 0 --sequential_edit 0 --bsz 100 --wrt 1
# # done
# echo >/jizhi/jizhi2/worker/trainer/.cfile0

# export CUDA_VISIBLE_DEVICES=1
# python Editing_baselines.py --method SERAC --model meta-llama/Llama-3.2-1B --dataset counterfact --batch_edit 1 --sequential_edit 0 --bsz 100 --wrt 0
# ds_array=("portability_inverse_relation" "portability_one_hop" "portability_subject_replace")
# for ds_str in "${ds_array[@]}"; do
#     python Editing_baselines.py --method MEMIT --model meta-llama/Llama-3.2-1B-Instruct --dataset $ds_str --batch_edit 1 --sequential_edit 0 --bsz 100 --wrt 1
# done

