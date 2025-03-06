export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
export ftp_proxy="http://star-proxy.oa.com:3128"

export TRANSFORMERS_CACHE="/apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/transformers_cache"
export HF_DATASETS_CACHE="/apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/transformers_cache"
export HF_HOME="/apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/transformers_cache"

# ./checkpoints/wikipara-Lamini-nli-MCQ-ZSRE-CF/checkpoints_2024-12-28_13-56/checkpoint_3500/

# portability_inverse_relation, portability_one_hop, portability_subject_replace



# echo clear 0 >~/.cfile
# python ./forward_check.py > ./logs/log_$(date +%Y-%m-%d_%H-%M-%S) 2>&1
# echo >~/.cfile


# eval
# export CUDA_VISIBLE_DEVICES=0
# echo clear 0 >/jizhi/jizhi2/worker/trainer/.cfile0
# python eval.py --method gist --model meta-llama/Llama-3.2-1B --dataset counterfact --batch_edit 1 --run_base 0 --bsz 500 --edits 3 --wrt 0 --weight_path ./checkpoints/wikipara-Lamini-nli-MCQ-ZSRE-CF/checkpoints_2025-01-25_16-13/checkpoint_1400/
# echo >/jizhi/jizhi2/worker/trainer/.cfile0

# export CUDA_VISIBLE_DEVICES=1
# python eval.py --method ICE --model meta-llama/Llama-3.2-1B --dataset counterfact --batch_edit 0 --run_base 0 --bsz 100 --edits 2 --wrt 1 --weight_path ./checkpoints/wikipara-Lamini-nli-MCQ-ZSRE-CF/checkpoints_2024-12-28_13-56/checkpoint_3500/


# Baseline
export CUDA_VISIBLE_DEVICES=0
echo clear 0 >/jizhi/jizhi2/worker/trainer/.cfile0
# python Editing_baselines_mquake.py --method GRACE --model meta-llama/Llama-3.2-1B --edits 2 --multiple_edit 0 --bsz 100 --wrt 0
for ii in {2..4}; do
    python Editing_baselines_mquake.py --method GRACE --model meta-llama/Llama-3.2-1B --edits $ii --multiple_edit 0 --bsz 100 --wrt 1
done
echo >/jizhi/jizhi2/worker/trainer/.cfile0


# export CUDA_VISIBLE_DEVICES=1
# # python Editing_baselines_mquake.py --method FT-M --model meta-llama/Llama-3.2-1B --edits 4 --multiple_edit 0 --bsz 100 --wrt 0
# # python lost_m_b.py --model meta-llama/Llama-3.2-1B --dataset counterfact --bsz 500 --cf_pos first --mq_pos 0 1 --wrt 0
# for ii in {2..4}; do
#     python Editing_baselines_mquake.py --method GRACE --model meta-llama/Llama-3.2-1B --edits $ii --multiple_edit 1 --bsz 100 --wrt 1
# done




# export CUDA_VISIBLE_DEVICES=0
# echo clear 0 >/jizhi/jizhi2/worker/trainer/.cfile0
# # python Editing_baselines.py --method SERAC --model meta-llama/Llama-3.2-1B --dataset zsre --batch_edit 1 --sequential_edit 0 --bsz 100 --wrt 1
# ds_array=("portability_inverse_relation" "portability_one_hop" "portability_subject_replace")
# for ds_str in "${ds_array[@]}"; do
#     python Editing_baselines.py --method GRACE --model meta-llama/Llama-3.2-1B --dataset $ds_str --batch_edit 0 --sequential_edit 1 --bsz 100 --wrt 1
# done
# echo >/jizhi/jizhi2/worker/trainer/.cfile0


# export CUDA_VISIBLE_DEVICES=1
# # python Editing_baselines.py --method GRACE --model meta-llama/Llama-3.2-1B --dataset zsre --batch_edit 0 --sequential_edit 1 --bsz 100 --wrt 0
# ds_array=("portability_inverse_relation" "portability_one_hop" "portability_subject_replace")
# for ds_str in "${ds_array[@]}"; do
#     python Editing_baselines.py --method GRACE --model meta-llama/Llama-3.2-1B --dataset $ds_str --batch_edit 0 --sequential_edit 0 --bsz 100 --wrt 1
# done



# export CUDA_VISIBLE_DEVICES=1
# # python lost_m_b.py --model meta-llama/Llama-3.2-1B --dataset counterfact --bsz 100 --cf_pos first --wrt 1
# ds_array=("first" "middle" "last")
# for ((i=100; i<=1000; i+=100)); do
#     for ds_str in "${ds_array[@]}"; do
#         python lost_m_b.py --model meta-llama/Llama-3.2-1B --dataset counterfact --bsz $i --cf_pos $ds_str --wrt 1
#     done
# done
