echo clear 0 1 2 3>/jizhi/jizhi2/worker/trainer/.cfile
python -m accelerate.commands.launch --num_processes=1 --mixed_precision=fp16 --gpu_ids=[0,1,2,3] /apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/edit_train.py
echo >/jizhi/jizhi2/worker/trainer/.cfile


# python /apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/edit_train.py