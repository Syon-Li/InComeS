from transformers import AutoTokenizer, LlamaForCausalLM, LlamaModel
from edit_attention import Edit_LlamaModel, Edit_LlamaForCausalLM
from utils import FileIterD, collate_fn_fileD, get_lr, get_num_lines
from accelerate import Accelerator, DataLoaderConfiguration
import argparse
import torch
import pickle
import json
import logging
import os
from datetime import datetime


# now = datetime.now()
# formatted_time = now.strftime('%Y-%m-%d_%H-%M-%S')

# logger = logging.getLogger(__file__)
# log_file = "log_{}".format(formatted_time)
# log_path = "logs/{}".format(log_file)

# fileHandler = logging.FileHandler(log_path, mode = 'w')
# fileHandler.setLevel(logging.DEBUG)
# consoleHandler = logging.StreamHandler()
# consoleHandler.setLevel(logging.DEBUG)

# # set formatter
# formatter = logging.Formatter('[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# consoleHandler.setFormatter(formatter)
# fileHandler.setFormatter(formatter)

# # add
# logger.addHandler(fileHandler)
# logger.addHandler(consoleHandler)

# logger.info("test")





def main():

    parser = argparse.ArgumentParser(description='Pretraining.')
    parser.add_argument('--gradaccu', type=int, default=1, help='number of gradient steps to use')
    parser.add_argument('--batch_size', type=int, nargs="+", default=[8,8,8,8], help='the batch size for each dataloader')
    parser.add_argument("--gist_num", type=int, default=10, help="number of gist token activations to keep")
    parser.add_argument('--num_gpu', type=int, default=1, help='number of gpus to use')
    args = parser.parse_args()

    lr = 4e-5
    min_lr = 4e-6

    warmup_updates = 1000
    max_updates = 20000
    save_updates = 2000

    warmup_updates_per_gpu = warmup_updates / args.num_gpu
    max_updates_per_gpu = max_updates / args.num_gpu
    save_updates_per_gpu = save_updates / args.num_gpu

    warmup_iters = int(warmup_updates_per_gpu * args.gradaccu)
    max_iters = int(max_updates_per_gpu * args.gradaccu)
    save_iters = int(save_updates_per_gpu * args.gradaccu)

    loss_f_name = "training_loss"
    if os.path.exists("{}.json".format(loss_f_name)):
        loss_f_name = loss_f_name + "_new"
    
    checkpoint_dir = "./checkpoints/slimpajama_10B"

    



    model = Edit_LlamaForCausalLM.from_pretrained("/apdcephfs_qy3/share_733425/zhisonzhang/zh/2401mygo/_cache/models--TinyLlama--TinyLlama-1.1B-intermediate-step-1431k-3T/snapshots/036fa4651240b9a1487f709833b9e4b96b4c1574")
    tokenizer = AutoTokenizer.from_pretrained("/apdcephfs_qy3/share_733425/zhisonzhang/zh/2401mygo/_cache/models--TinyLlama--TinyLlama-1.1B-intermediate-step-1431k-3T/snapshots/036fa4651240b9a1487f709833b9e4b96b4c1574")
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
        gist_pool.update({i: 
                        {"keys": torch.zeros((1, model.config.num_key_value_heads, model.config.hidden_size // model.config.num_attention_heads)), 
                        "values": torch.randn((1, model.config.num_key_value_heads, model.config.hidden_size // model.config.num_attention_heads))}
                        })
        

    
    Slice = ["0_256","256_512","512_1024","1024_inf"]
    # Slice = ["0_256","256_512"]

    
    f_path_set, start_set, end_set = [], [], []
    for i,s in enumerate(Slice[:]):
        f_path = "./slimpajama_sample/slimpajama_train_{}_new.json".format(s)
        # num_lines = get_num_lines(f_path)
        # print("num_lines", num_lines)
        f_path_set.append(f_path)
        start_set.append(0)
        end_set.append(2048)

    ds = FileIterD(f_path_set=f_path_set, start_set=start_set, end_set=end_set, batch_size=args.batch_size)
    train_dataloader = torch.utils.data.DataLoader(ds, collate_fn=collate_fn_fileD)
    # print(list(train_dataloader)[:3])


    accelerator = Accelerator(
                            gradient_accumulation_steps=args.gradaccu, 
                            dataloader_config=DataLoaderConfiguration(dispatch_batches=False), 
                            )
    print(accelerator.device, accelerator.process_index, accelerator.distributed_type,)
    # model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    loss_weights = {"sparsity_loss_w":1,"p0_loss_w":1,"pS_loss_w":1}
    


    # accelerator.load_state(input_dir="/apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/Gist_slimpajama_checkpoint")
    # print(accelerator.state)
    model.train()
    for local_step, (global_step, input_ids, attention_mask, labels) in enumerate(train_dataloader):
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
            
            print(input_ids, input_ids.shape, local_step+1, global_step+1)
            # input_ids = input_ids.to(device)
            # attention_mask = attention_mask.to(device)
            outputs, gist_pool, loss_set = model(input_ids=input_ids, 
                                                attention_mask=attention_mask, 
                                                labels=labels,
                                                gist_pool=gist_pool, 
                                                gist_pool_idx=gist_pool_idx, 
                                                gist_token_ids=gist_token_ids,
                                                loss_weights=loss_weights,
                                                use_cache=False,
                                                )
            loss = outputs.loss
            # print(loss_set)
            accelerator.backward(loss)
            optimizer.step()
            lr_updated = get_lr(global_step+1, warmup_iters=warmup_iters, lr_decay_iters=max_iters, min_lr=min_lr, learning_rate=lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_updated
                # print("it;lr",it, param_group["lr"])
            optimizer.zero_grad()

        loss_set.update({"global_step":global_step+1, "local_step":local_step+1})
        with open("{}.json".format(loss_f_name), "a") as file:
            json.dump(loss_set, file)
            file.write("\n")

        for key,value in gist_pool.items():
            value["keys"] = value["keys"].detach()
            value["values"] = value["values"].detach()
            if len(value["keys"]) > args.gist_num:
                value["keys"] = value["keys"][-args.gist_num:]
                value["values"] = value["values"][-args.gist_num:]
            # print("value[keys].shape[0]", value["keys"].shape[0])
        
        if (local_step+1) % save_iters == 0:
            accelerator.wait_for_everyone()
            accelerator.save_state(output_dir="{}/checkpoint_{}".format(checkpoint_dir, local_step+1))
            if accelerator.is_main_process:
                checkpoint_config = {"gist_pool":gist_pool, "global_step":global_step+1, "local_step":local_step+1}
                with open('{}/checkpoint_{}/checkpoint_config.pickle'.format(checkpoint_dir, local_step+1), 'wb') as f:
                    pickle.dump(checkpoint_config, f)

     

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained("{}/model".format(checkpoint_dir), is_main_process=accelerator.is_main_process, save_function=accelerator.save,)
            




if __name__=="__main__":
    main()

