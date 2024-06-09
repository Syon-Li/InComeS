from transformers import AutoTokenizer, LlamaForCausalLM, LlamaModel
from edit_attention import Edit_LlamaModel, Edit_LlamaForCausalLM
from utils import ArrIterD, FileIterD, collate_fn_fileD, get_lr
from accelerate import Accelerator, DataLoaderConfiguration
import numpy as np
import argparse
import torch
import random
import pickle




def main():

    parser = argparse.ArgumentParser(description='Pretraining.')
    parser.add_argument('--gradaccu', type=int, default=1, help='number of gradient steps to use')
    parser.add_argument('--batch_size', type=int, nargs="+", default=[1,1,1,1,1], help='the batch size for each dataloader')
    parser.add_argument('--epoch', type=int, default=1, help='the epoch number to train')
    parser.add_argument("--gist_num", type=int, default=10, help="number of gist token activations to keep")
    parser.add_argument('--slice', type=int, nargs="+", help='the slice in the data')
    args = parser.parse_args()

    lr = 4e-5
    min_lr = 4e-06

    warmup_updates = 200
    max_updates = 20000
    save_updates = 5

    warmup_iters = warmup_updates * args.gradaccu
    max_iters = max_updates * args.gradaccu
    lr_decay_iters = max_iters
    save_iters = save_updates * args.gradaccu




    

    model = Edit_LlamaForCausalLM.from_pretrained("/apdcephfs_qy3/share_733425/zhisonzhang/zh/2401mygo/_cache/models--TinyLlama--TinyLlama-1.1B-intermediate-step-1431k-3T/snapshots/036fa4651240b9a1487f709833b9e4b96b4c1574")
    tokenizer = AutoTokenizer.from_pretrained("/apdcephfs_qy3/share_733425/zhisonzhang/zh/2401mygo/_cache/models--TinyLlama--TinyLlama-1.1B-intermediate-step-1431k-3T/snapshots/036fa4651240b9a1487f709833b9e4b96b4c1574")
    # print(model)


    tokenizer.pad_token = tokenizer.eos_token
    num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens": ["<GIST>"]})
    model.resize_token_embeddings(len(tokenizer))
    print(tokenizer.additional_special_tokens, tokenizer.additional_special_tokens_ids)
    gist_token_ids = tokenizer.additional_special_tokens_ids[-1]

    with torch.no_grad():
        model.model.embed_tokens.weight[-1] = model.model.embed_tokens.weight[:-1].mean(0)
        model.lm_head.weight[-1] = model.lm_head.weight[:-1].mean(0)



    # arr = np.memmap("/apdcephfs_qy3/share_733425/zhisonzhang/zh/2401mygo/data3/SlimPajama-100B-sample/slimPv0.T63.bin", dtype=np.uint16, mode='r')
    # print(arr[:10], arr.shape, type(arr[:10]))

    # ds = ArrIterD(mem_arr=arr[:10000], 
    #         gist_token_ids=tokenizer.convert_tokens_to_ids("<GIST>"),
    #         gist_location_id=tokenizer.convert_tokens_to_ids("."), 
    #         config=model.config)




    gist_pool = {}
    for i in range(model.config.num_hidden_layers):
        gist_pool.update({i: 
                        {"keys": torch.zeros((1, model.config.num_key_value_heads, model.config.hidden_size // model.config.num_attention_heads)), 
                        "values": torch.zeros((1, model.config.num_key_value_heads, model.config.hidden_size // model.config.num_attention_heads))}
                        })
        


    
    Slice = ["0_256","256_512","512_1024","1024_2048","2048_inf"]
    # Slice = ["0_256","256_512","512_1024"]

    
    datasets = []
    for i,S in enumerate(Slice):
        f_path = "/apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/slimpajama_train_{}.json".format(S)
        with open(f_path, "r") as f:
            lines = f.readlines()
            # print('lines of the file', len(lines))
        ds = FileIterD(lines=lines, start=0, end=16, batch_size=args.batch_size[i])
        datasets.append(ds)
    
    datasets = torch.utils.data.ChainDataset(datasets)
    train_dataloader = torch.utils.data.DataLoader(datasets, collate_fn=collate_fn_fileD)
    # print(list(train_dataloader)[:3])


    dataloader_config = DataLoaderConfiguration(dispatch_batches=False)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradaccu, dataloader_config=dataloader_config)
    device = accelerator.device
    print(device)
    # model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    loss_weights = (1e-3,1e-3,1e-2)
    loss_weights = {"sparsity_loss_w":1,"p0_loss_w":1e-3,"pS_loss_w":1e-1}


    # accelerator.load_state(input_dir="/apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/Gist_slimpajama_checkpoint")
    # print(accelerator.state)
    model.train()
    for it, (input_ids, attention_mask, labels) in enumerate(train_dataloader):
        with accelerator.accumulate(model):
            bsz = input_ids.shape[0]
            gist_pool_idx = torch.zeros(bsz, len(gist_pool[0]["keys"])+1) #plus one to consider the zero gist key and value
            gist_pool_idx = torch.concat([gist_pool_idx, torch.eye(bsz)], dim=1)     
            # print(gist_pool_idx, gist_pool_idx.shape)
            print(input_ids, input_ids.shape)
            # input_ids = input_ids.to(device)
            # attention_mask = attention_mask.to(device)
            optimizer.zero_grad()
            outputs, gist_pool = model(input_ids=input_ids, 
                                    attention_mask=attention_mask, 
                                    labels=labels, 
                                    gist_pool=gist_pool, 
                                    gist_pool_idx=gist_pool_idx, 
                                    gist_token_ids=gist_token_ids,
                                    loss_weights=loss_weights,
                                    use_cache=False,
                                    )
            loss = outputs.loss
            print(loss)
            lr = get_lr(it, warmup_iters=4, lr_decay_iters=lr_decay_iters, min_lr=min_lr, learning_rate=lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
                print("it;lr",it, param_group["lr"])
            accelerator.backward(loss)
            optimizer.step()

        for key,value in gist_pool.items():
            value["keys"] = value["keys"].detach()
            value["values"] = value["keys"].detach()
            if len(value["keys"]) > args.gist_num:
                value["keys"] = value["keys"][-args.gist_num:]
                value["values"] = value["values"][-args.gist_num:]
            # print("value[keys].shape[0]", value["keys"].shape[0])
        
        if (it+1) % save_iters == 0:
            accelerator.save_state(output_dir="/apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/Gist_slimpajama_checkpoint/checkpoint_{}".format(it))
            checkpoint_config = {"gist_pool":gist_pool, "iteration":it}
            with open('/apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/Gist_slimpajama_checkpoint/checkpoint_{}/checkpoint_config.pickle'.format(it), 'wb') as f:
                pickle.dump(checkpoint_config, f)

            




if __name__=="__main__":
    main()

