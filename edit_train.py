from transformers import AutoTokenizer, LlamaForCausalLM, LlamaModel
from edit_attention import Edit_LlamaModel, Edit_LlamaForCausalLM
from utils import ArrIterD, FileIterD, collate_fn_fileD, CustomBatchSampler
from accelerate import Accelerator
import numpy as np
import argparse
import torch
import random




def main():

    parser = argparse.ArgumentParser(description='Pretraining.')
    parser.add_argument('--num_workers', type=int, help='number of GPUs to use')
    parser.add_argument('--batch_size', type=int, nargs="+", help='the batch size for each dataloader')
    parser.add_argument('--epoch', type=int, help='the epoch number to train')
    parser.add_argument("--gist_num", type=int, help="number of gist token activations to keep")
    parser.add_argument('--slice', type=int, nargs="+", help='the slice in the data')
    args = parser.parse_args()



    # model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    # model = LlamaModel.from_pretrained("meta-llama/Llama-2-7b-hf")

    # model = Edit_LlamaModel.from_pretrained("meta-llama/Llama-2-7b-hf")
    # model = Edit_LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    # model = Edit_LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T").to("cuda")
    # tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")

    model = Edit_LlamaForCausalLM.from_pretrained("/apdcephfs_qy3/share_733425/zhisonzhang/zh/2401mygo/_cache/models--TinyLlama--TinyLlama-1.1B-intermediate-step-1431k-3T/snapshots/036fa4651240b9a1487f709833b9e4b96b4c1574")
    tokenizer = AutoTokenizer.from_pretrained("/apdcephfs_qy3/share_733425/zhisonzhang/zh/2401mygo/_cache/models--TinyLlama--TinyLlama-1.1B-intermediate-step-1431k-3T/snapshots/036fa4651240b9a1487f709833b9e4b96b4c1574")


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




    gist_activations = {}
    for i in range(model.config.num_hidden_layers):
        gist_activations.update({i: 
                                {"keys": torch.randn((1, model.config.num_key_value_heads, model.config.hidden_size // model.config.num_attention_heads), requires_grad=True), 
                                "values": torch.zeros((1, model.config.num_key_value_heads, model.config.hidden_size // model.config.num_attention_heads))}
                                })


    
    Slice = ["0_256","256_512","512_1024","1024_2048","2048_inf"]

    
    datasets = []
    for i,S in enumerate(Slice):
        f_path = "/apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/slimpajama_train_{}.json".format(S)
        with open(f_path, "r") as f:
            lines = f.readlines()
            print('lines of the file', len(lines))
        ds = FileIterD(lines=lines, start=0, end=10, batch_size=args.batch_size[i])
        datasets.append(ds)
    
    datasets = torch.utils.data.ChainDataset(datasets)
    train_dataloader = torch.utils.data.DataLoader(datasets, num_workers=args.num_workers, collate_fn=collate_fn_fileD,)
    # print(list(train_dataloader)[:3])


    accelerator = Accelerator()
    device = accelerator.device
    # model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    model, optimizer, scheduler, train_dataloader = accelerator.prepare(model, optimizer, scheduler, train_dataloader)
    loss_weights = (1,1,1)


    model.train()

    for input_ids, attention_mask, labels in train_dataloader:
        gist_pool_idx = torch.zeros(input_ids.shape[0], gist_activations[0]["keys"].shape[0])
        gist_pool_idx = torch.concat([gist_pool_idx, torch.eye(input_ids.shape[0])], dim=1)     
        # print(gist_pool_idx, gist_pool_idx.shape)
        print(input_ids, input_ids.shape)
        # input_ids = input_ids.to(device)
        # attention_mask = attention_mask.to(device)
        optimizer.zero_grad()
        outputs, gist_activations = model(input_ids=input_ids, 
                                          attention_mask=attention_mask, 
                                          labels=labels, 
                                          gist_activations=gist_activations, 
                                          gist_pool_idx=gist_pool_idx, 
                                          gist_token_ids=gist_token_ids,
                                          loss_weights=loss_weights,
                                          use_cache=False,
                                          )
        loss = outputs.loss
        print(loss)
        accelerator.backward(loss.sum())
        # loss.backward()
        optimizer.step()
        scheduler.step()

        for key,value in gist_activations.items():
            value["keys"] = value["keys"].detach()
            value["values"] = value["values"].detach()
            if value["keys"].shape[0] > args.gist_num:
                value["keys"] = value["keys"][-args.gist_num:]
                value["values"] = value["values"][-args.gist_num:]
            




if __name__=="__main__":
    main()

