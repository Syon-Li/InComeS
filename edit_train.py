from transformers import AutoTokenizer, LlamaForCausalLM, LlamaModel
from edit_attention import Edit_LlamaModel, Edit_LlamaForCausalLM
from utils import FileIterD, collate_fn_fileD, get_lr, get_num_lines, wrap_collate_fn, HFIterD, collate_fn_hf
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import GradientAccumulationPlugin
import argparse
import torch
import pickle
import json
import os
import datetime
from datasets import load_dataset

os.environ["NCCL_DEBUG"] = "INFO"

now = datetime.datetime.now()


def main():

    parser = argparse.ArgumentParser(description='Pretraining.')
    parser.add_argument('--gradaccu', type=int, default=8, help='number of gradient steps to accumulate')
    parser.add_argument('--batch_size', type=int, nargs="+", default=[16,8,4,4], help='the batch size for each dataloader')
    parser.add_argument("--gist_num", type=int, default=100, help="number of gist token activations to keep")
    args = parser.parse_args()

    start_point = 0
    end_point = 2

    lr = 2e-5
    min_lr = 2e-6

    warmup_updates = 1000
    max_updates = 20000
    save_updates = 20

    # warmup_updates = 50
    # max_updates = 500
    # save_updates = 100

    accu_num = args.gradaccu

    warmup_iters = warmup_updates * accu_num
    max_iters = max_updates * accu_num
    save_iters = save_updates * accu_num
    
    model_name = "TinyLlama-1.1B-Chat"

    loss_f_name = "train_loss_{}_{}".format(model_name, now.strftime('%Y-%m-%d_%H-%M'))

    
    batch_size = args.batch_size
    # batch_size = 8

    gist_num = args.gist_num

    ds_name = "slimpajama_10B"
    # ds_name = "openai_gsm8k"
    extra_info = "no bias; trained without extra loss"

    checkpoint_dir = "./checkpoints/{}/checkpoints_{}".format(ds_name, now.strftime('%Y-%m-%d_%H-%M'))

    



    # model = Edit_LlamaForCausalLM.from_pretrained("/apdcephfs_qy3/share_733425/zhisonzhang/zh/2401mygo/_cache/models--TinyLlama--TinyLlama-1.1B-intermediate-step-1431k-3T/snapshots/036fa4651240b9a1487f709833b9e4b96b4c1574")
    # tokenizer = AutoTokenizer.from_pretrained("/apdcephfs_qy3/share_733425/zhisonzhang/zh/2401mygo/_cache/models--TinyLlama--TinyLlama-1.1B-intermediate-step-1431k-3T/snapshots/036fa4651240b9a1487f709833b9e4b96b4c1574")

    model = Edit_LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", cache_dir="transformers_cache")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", cache_dir="transformers_cache")
    # print(model)


    tokenizer.pad_token = tokenizer.eos_token
    num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens": ["<GIST>"]})
    model.resize_token_embeddings(len(tokenizer))
    # print(tokenizer.additional_special_tokens, tokenizer.additional_special_tokens_ids)
    gist_token_ids = tokenizer.additional_special_tokens_ids[-1]

    with torch.no_grad():
        model.model.embed_tokens.weight[-1] = model.model.embed_tokens.weight[:-1].mean(0)
        model.lm_head.weight[-1] = model.lm_head.weight[:-1].mean(0)


    gist_pool = {}
    for i in range(model.config.num_hidden_layers):
        gist_pool.update({i:{"keys":torch.tensor([]), "values":torch.tensor([])}})
        

    
    Slice = ["0_256","256_512","512_1024","1024_inf"]
    # Slice = ["0_256","256_512"]

    
    f_path_set, start_set, end_set = [], [], []
    for i,s in enumerate(Slice[:]):
        f_path = "./slimpajama_sample/slimpajama_train_{}_new.json".format(s)
        # num_lines = get_num_lines(f_path)
        # print("num_lines", num_lines)
        f_path_set.append(f_path)
        start_set.append(start_point)
        end_set.append(int(end_point))

    ds = FileIterD(f_path_set=f_path_set, start_set=start_set, end_set=end_set, batch_size=batch_size)
    train_dataloader = torch.utils.data.DataLoader(ds, collate_fn=collate_fn_fileD)
    # print(list(train_dataloader)[:3])


    # ds = HFIterD(f_path="openai/gsm8k", batch_size=batch_size, subset="main", split="train", cache_dir="transformers_cache", tokenizer=tokenizer)
    # ds = load_dataset("openai/gsm8k", "main", split="train", cache_dir="transformers_cache", streaming=True)
    # train_dataloader = torch.utils.data.DataLoader(ds, collate_fn=collate_fn_hf)
    # for i, (input_ids, attention_mask, labels) in enumerate(train_dataloader):
    #     print(input_ids, input_ids.shape)
    #     print(attention_mask, attention_mask.shape)
    #     print(labels, labels.shape)
    #     if i==5:
    #         break



    accelerator = Accelerator(
                            dataloader_config=DataLoaderConfiguration(dispatch_batches=False), 
                            gradient_accumulation_plugin=GradientAccumulationPlugin(num_steps=accu_num, 
                                                                                    #   sync_each_batch=True,
                                                                                    ),
                            log_with="wandb",
                            )
    print(accelerator.device, accelerator.process_index, accelerator.distributed_type,)
    # model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # print("warmup_iters: ", warmup_iters)
    scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/warmup_updates, total_iters=warmup_iters)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(max_iters - warmup_iters), eta_min=min_lr)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_iters])
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, scheduler)
    loss_weights = {"sparsity_loss_w":1e-2, "p0_loss_w":1e-2, "pS_loss_w":1e-2}
    


    # accelerator.load_state(input_dir="/apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/checkpoints/slimpajama_10B/checkpoints_2024-07-12_23-00/checkpoint_32000")
    # with open("/apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/checkpoints/slimpajama_10B/checkpoints_2024-07-12_23-00/checkpoint_48000/checkpoint_config.pickle", 'rb') as f:
    #     obj = pickle.load(f)
    # gist_pool = obj["gist_pool"] 
    # print("gist_values", gist_pool[2]["values"], gist_pool[2]["values"].shape)

    config = {"model": model_name,
              "batch_size": batch_size,
              "accu_num": accu_num,
              "gist_num": gist_num,
              "loss_weights": loss_weights,
              "extra_info": extra_info,
              "dataset": ds_name,}
    # accelerator.init_trackers("gist-model-editing", config=config)
    model.train()
    for local_step, (input_ids, attention_mask, labels) in enumerate(train_dataloader):
        with accelerator.accumulate(model):
            bsz = input_ids.shape[0]
            gist_pool_idx = torch.zeros(bsz, len(gist_pool[0]["keys"])+1) #plus one to consider the zero gist key and value
            # gist_pool_idx = torch.concat([gist_pool_idx, torch.eye(bsz)], dim=1)  
            gist_pool_idx = torch.concat([gist_pool_idx, torch.zeros(bsz, bsz)], dim=1)  
            col = len(gist_pool[0]["keys"])+1
            for i in range(gist_pool_idx.shape[0]):
                if gist_token_ids in input_ids[i]:
                    gist_pool_idx[i,col] = 1
                    col += 1
            # print("gist_pool_idx ", gist_pool_idx, gist_pool_idx.shape)
            outputs, gist_pool, loss_set = model(input_ids=input_ids, 
                                                attention_mask=attention_mask, 
                                                labels=labels,
                                                gist_pool=gist_pool, 
                                                gist_pool_idx=gist_pool_idx, 
                                                gist_token_ids=gist_token_ids,
                                                loss_weights=loss_weights,
                                                use_cache=False,
                                                )
            # loss = outputs.loss
            # accelerator.backward(loss)
            # optimizer.step()
            # scheduler.step()
            # optimizer.zero_grad()
            # print('{}, local_step:{}, lr:{}, updates:{}'.format(input_ids.shape, local_step+1, optimizer.param_groups[0]['lr'], (local_step+1) // accu_num)) 

        # if (local_step+1) % accu_num == 0:
        #     updates = (local_step+1) // accu_num
        #     loss_set.update({"lr":optimizer.param_groups[0]['lr']})
        #     accelerator.log(loss_set, step=updates)
        #     # print(loss_set, updates)
        #     if updates % save_updates==0:
        #         accelerator.wait_for_everyone()
        #         accelerator.save_state(output_dir="{}/checkpoint_{}".format(checkpoint_dir, local_step+1))
        #         if accelerator.is_main_process:
        #             checkpoint_config = {"gist_pool":gist_pool, "local_step": local_step+1}
        #             with open('{}/checkpoint_{}/checkpoint_config.pickle'.format(checkpoint_dir, local_step+1), 'wb') as f:
        #                 pickle.dump(checkpoint_config, f)                    

        #     loss_set.update({"input_ids_shape":input_ids.shape, "local_step": local_step+1, "lr":optimizer.param_groups[0]['lr'], "updates": updates})
        #     with open("loss_record/{}.json".format(loss_f_name), "a") as file:
        #         json.dump(loss_set, file)
        #         file.write("\n")

        for key,value in gist_pool.items():
            value["keys"] = value["keys"].detach()
            value["values"] = value["values"].detach()
            if len(value["keys"]) > gist_num:
                value["keys"] = value["keys"][-gist_num:]
                value["values"] = value["values"][-gist_num:]
            # print("value[keys].shape[0]", value["keys"].shape[0])


    # accelerator.wait_for_everyone()
    # accelerator.save_model(model, "{}/model".format(checkpoint_dir))  
    # accelerator.end_training() 
    
            




if __name__=="__main__":
    main()

