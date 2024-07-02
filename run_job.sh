# echo clear 0 >/jizhi/jizhi2/worker/trainer/.cfile0
# python -m accelerate.commands.launch --num_processes=1 --mixed_precision=fp16 --gpu_ids=0 /apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/edit_train.py --gradaccu 1 --batch_size 16 8 4 4 --gist_num 100 > log.testing
# echo >/jizhi/jizhi2/worker/trainer/.cfile0

# for ii in 0 1 2 3 4 5 6 7; do echo clear ${ii} >/jizhi/jizhi2/worker/trainer/.cfile${ii}; done
# python -m accelerate.commands.launch --config_file /apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/fsdp_config.yaml /apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/edit_train.py --gradaccu 8 --num_gpu 8 --batch_size 32 16 8 8 --gist_num 100 > log.testing &
# for ii in 0 1 2 3 4 5 6 7; do echo >/jizhi/jizhi2/worker/trainer/.cfile${ii}; done


python -m accelerate.commands.launch --config_file /apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/fsdp_config.yaml /apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/edit_train.py --gradaccu 16 --batch_size 16 8 4 4 --gist_num 100 > logs/log_$(date +%Y-%m-%d_%H-%M-%S) 2>&1


# python data_prep.py --proportion 0.016 --input_ffile /apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/slimpajama_train_file.json --split 64 --split_i 4

# python --mode ffile

# for i in {0..199}; do
# python data_prep.py --mode tokenize --proportion 0.016 --input_ffile /apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/slimpajama_train_file.json --split 200 --split_i ${i} > log.s${i} &
# done


# ps aux | grep python
