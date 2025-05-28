export TRANSFORMERS_CACHE="your/transformers_cache"
export HF_DATASETS_CACHE="your/transformers_cache"
export HF_HOME="your/transformers_cache"

# eval
export CUDA_VISIBLE_DEVICES=1
python eval.py --method gist --model meta-llama/Llama-3.2-1B --dataset mquake --batch_edit 0 --run_base 0 --bsz 100 --edits 2 --wrt 0 --weight_path ./checkpoints/wikipara-Lamini-nli-MCQ-ZSRE-CF/checkpoints_2025-05-11_15-44/ --checkpoint_tag model
