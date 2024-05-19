from transformers import AutoTokenizer, LlamaForCausalLM, LlamaModel
from edit_attention import Edit_LlamaModel, Edit_LlamaForCausalLM
from utils import MyIterableDataset, chunk_arr
from datasets import load_dataset
from accelerate import Accelerator

from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer
import numpy as np
import torch




def main():
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
    num_added_toks = tokenizer.add_special_tokens({"cls_token": "<GIST>"})
    model.resize_token_embeddings(len(tokenizer))
    # print(tokenizer.additional_special_tokens, tokenizer.additional_special_tokens_ids)
    print(tokenizer.convert_tokens_to_ids("."))
    # print(tokenizer.encode("This is an apple. That is a person."))


    arr = np.memmap("/apdcephfs_qy3/share_733425/zhisonzhang/zh/2401mygo/data3/SlimPajama-100B-sample/slimPv0.T63.bin", dtype=np.uint16, mode='r')
    print(arr[:10], arr.shape, type(arr[:10]))



    gist_activations = {}
    for i in range(model.config.num_hidden_layers):
        gist_activations.update({i: {"keys": torch.randn((20, model.config.num_key_value_heads, model.config.hidden_size // model.config.num_attention_heads)), 
                                     "values": torch.randn((20, model.config.num_key_value_heads, model.config.hidden_size // model.config.num_attention_heads))}})
    
    
    # gist_pool_idx = torch.zeros(gist_activations[0]["keys"].shape[0]+len(prompt)).to("cuda")
    # gist_pool_idx[-len(prompt):] = 1
    # outputs, gist_activations = model(inputs["input_ids"], inputs["attention_mask"], gist_activations=gist_activations, gist_pool_idx=gist_pool_idx, gist_token_ids=tokenizer.convert_tokens_to_ids(tokenizer.cls_token))


    ds = MyIterableDataset(mem_arr=arr[:10000], 
                           gist_activations=gist_activations, 
                           gist_token_ids=tokenizer.convert_tokens_to_ids("<GIST>"),
                           gist_location_id=tokenizer.convert_tokens_to_ids("."), 
                           config=model.config)
    # print(list(torch.utils.data.DataLoader(ds, num_workers=2))[:3])
    training_dataloader = torch.utils.data.DataLoader(ds, num_workers=4, batch_size=4)

    accelerator = Accelerator()
    device = accelerator.device
    # model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    model, optimizer, training_dataloader, scheduler = accelerator.prepare(model, optimizer, training_dataloader, scheduler)
    gist_token_ids = 32000


    model.train()
    for input_ids, attention_mask, labels in training_dataloader:
        gist_pool_idx = torch.zeros(input_ids.shape[0], gist_activations[0]["keys"].shape[0])
        gist_pool_idx = torch.hstack([gist_pool_idx, torch.eye(input_ids.shape[0])])     
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
                                          use_cache=False,
                                          )
        loss = outputs.loss
        print(loss)
        accelerator.backward(loss)
        # loss.backward()
        optimizer.step()
        scheduler.step()





if __name__=="__main__":
    main()

