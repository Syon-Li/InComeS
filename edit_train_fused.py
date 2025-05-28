from transformers import AutoTokenizer, AutoModelForCausalLM
from edit_attention_llama import Edit_LlamaForCausalLM
# from edit_attention_gptj import Edit_GPTJForCausalLM
from edit_attention_qwen import Edit_Qwen2ForCausalLM
from utils import wrap_collate_fn, HFIterD, mask_gist, mask_context, reverse_cumsum, \
collate_fn, padding_fn, _chunks, get_lr, set_seed
from accelerate import Accelerator, DataLoaderConfiguration, DeepSpeedPlugin
from accelerate.utils import get_active_deepspeed_plugin, broadcast_object_list, DummyOptim
from liger_kernel.transformers import apply_liger_kernel_to_llama, LigerFusedLinearCrossEntropyLoss, apply_liger_kernel_to_qwen2
import torch.nn.functional as F
import argparse
import torch
import random
import math
import os
import datetime
import wandb




os.environ["NCCL_DEBUG"] = "INFO"

now = datetime.datetime.now()

torch.set_printoptions(threshold=float('inf'))




def main():

    set_seed(42)

    parser = argparse.ArgumentParser(description='Pretraining.')
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-1B", help='the model to use')
    parser.add_argument('--gist_bsz', type=int, default=256, help='processing gist batch size')
    args = parser.parse_args()

    batch_size = args.gist_bsz
    model_path = args.model
    model_name = model_path[model_path.rfind("/")+1:]

    start_point = 0
    end_point = None

    if "qwen2.5-7b" in model_path.lower():
        lr = 5e-6
    else:
        lr = 1e-5
    min_lr = 1e-6

    beta = 0.5

    max_updates = 3000
    if "qwen2.5-7b" in model_path.lower():
        # warmup_updates = 150
        warmup_updates = 300
    elif "qwen2.5-3b" in model_path.lower():
        warmup_updates = 150
    else:
        warmup_updates = 300
    save_updates = max_updates // 5

    
    ds_name = "wikipara-Lamini-nli-MCQ-ZSRE-CF"
    extra_info = ""

    checkpoint_dir = "./checkpoints/{}/checkpoints_{}".format(ds_name, now.strftime('%Y-%m-%d_%H-%M'))

    

    if "llama" in model_path.lower():
        apply_liger_kernel_to_llama()
        model = Edit_LlamaForCausalLM.from_pretrained(model_path, local_files_only=True, attn_implementation="flash_attention_2", 
                                                      torch_dtype=torch.bfloat16)
    elif "qwen" in model_path.lower():
        apply_liger_kernel_to_qwen2()
        model = Edit_Qwen2ForCausalLM.from_pretrained(model_path, local_files_only=True, attn_implementation="flash_attention_2", 
                                                      torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable({"use_reentrant":False})
    origin_model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, attn_implementation="flash_attention_2", 
                                                        torch_dtype=torch.bfloat16)
    
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens": ["<GIST>"]})
    model.resize_token_embeddings(len(tokenizer))
    # print(tokenizer.pad_token_id, tokenizer.additional_special_tokens_ids)
    gist_token_ids = tokenizer.additional_special_tokens_ids[-1]
    print("bos_token_id", tokenizer.bos_token_id)
    print("pad_token_id", tokenizer.pad_token_id)


    
    with torch.no_grad():
        model.model.embed_tokens.weight[-1] = model.model.embed_tokens.weight[:-1].mean(0)
        model.lm_head.weight[-1] = model.lm_head.weight[:-1].mean(0)

 

    gist_pool = {}
    for i in range(model.config.num_hidden_layers):
        gist_pool.update({i:{"keys":torch.tensor([]), "values":torch.tensor([])}})

    

    deepspeed_plugins = {"student":DeepSpeedPlugin(hf_ds_config="./zero_stage3_config.json",), 
                        "teacher":DeepSpeedPlugin(offload_param_device="cpu", offload_optimizer_device="cpu", zero_stage=3),}
    accelerator = Accelerator(
                            dataloader_config=DataLoaderConfiguration(dispatch_batches=False,), 
                            deepspeed_plugin=deepspeed_plugins,
                            log_with="wandb",
                            )

    
    # Broadcast the data from process 0 to all other processes
    broadcast_object_list([checkpoint_dir], from_process=0)
    print(accelerator.device, accelerator.process_index, accelerator.distributed_type, accelerator.num_processes, checkpoint_dir)



    hf_f_path = ["sentence-transformers/s2orc", "sentence-transformers/agnews",
                "rajpurkar/squad", "LLukas22/nq-simplified", "allenai/openbookqa", "allenai/qasc", 
                 "openlifescienceai/medmcqa", "NASP/neteval-exam"]
    subset = ["title-abstract-pair", None, None, None, "main", None, None, None]

    f_path = ["./Editing_data/zsre/zsre_mend_train.json", "./Editing_data/counterfact/counterfact-train.json", 
              "./KnowEdit/benchmark/wiki_counterfact/train_cf.json"]

    weights = [4.5, 1.15, 0.086, 0.0909, 0.00596, 0.00813, 0.183, 0.005269, 0.49, 0.03, 0.008]
    split = ["train", "train", "train", "train", "train", "train", "train", "train"]
    assert len(hf_f_path)==len(split) and len(hf_f_path)==len(subset), "length of hf_f_path does not match subset or split"


    ds = HFIterD(hf_f_path=hf_f_path, f_path=f_path, subset=subset, 
                 split=split, tokenizer=tokenizer, weights=weights, 
                 lines=end_point)
    train_dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = DummyOptim(model.parameters(), lr=lr)
    
    

    if "llama" in model_path.lower():
        T_lm_head_w = origin_model.model.embed_tokens.weight.data.clone()
        T_lm_head_w = torch.concat([T_lm_head_w, torch.zeros_like(T_lm_head_w[-1][None,...])], dim=0).detach()

        txt_kn_csz = [(8,8), (16,16), (32,32), (64,64), (128,128)]
        gist_bsz_w = [0.05, 0.05, 0.05, 0.15, 0.7]

    elif "qwen" in model_path.lower():
        T_lm_head_w = origin_model.lm_head.weight
        T_lm_head_w = torch.concat([T_lm_head_w, torch.zeros_like(T_lm_head_w[-1][None,...])], dim=0).detach()

        txt_kn_csz = [(8,8), (16,16), (32,32), (64,64), (128,128)]
        gist_bsz_w = [0.05, 0.05, 0.05, 0.15, 0.7]



    active_plugin = get_active_deepspeed_plugin(accelerator.state)
    assert active_plugin is deepspeed_plugins["student"], "student deepspeed plugin was not activated"
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    accelerator.state.select_deepspeed_plugin("teacher")
    T_model, _ = accelerator.prepare(origin_model.model, train_dataloader)

    fused_CE = LigerFusedLinearCrossEntropyLoss(reduction="none")



    config = {"model": model_name,
              "save_updates": save_updates,
              "warmup_updates": warmup_updates,
              "hf_f_path": hf_f_path,
              "f_path": f_path,
              "sample_weights": weights,
              "batch_size": batch_size,
              "txt_kn_bsz": txt_kn_csz,
              "txt_bsz_weight": gist_bsz_w,
              "extra_info": extra_info,
              "beta": beta,
              "dataset": ds_name,
              "extra_info": extra_info}
    accelerator.init_trackers("gist-model-editing", config=config)
    

    model.train()
    T_model.eval()
    T_lm_head_w = T_lm_head_w.to(accelerator.device)
    rd_extra_loss = {"CrossEntropy_loss": [], "Weighted_CrossEntropy_loss":[], "kl_loss":[]}
    base_model = model.model
    for local_step, (kn, txt, kn_txt, txt_stem) in enumerate(train_dataloader):          
        # print("kn", kn, sep="\n")
        # print("txt", txt, sep="\n")
        # print("kn_txt", kn_txt, sep="\n")

        kn_cnt = 0
        gradient_accumulation_steps = 0
        chunk_scheme = []
        while kn_cnt < len(kn):
            kn_csz, txt_csz = random.choices(txt_kn_csz, weights=gist_bsz_w, k=1)[0]
            chunk_scheme.append((kn_csz, txt_csz))
            kn_cnt += kn_csz
            if kn_cnt <= len(kn):
                gradient_accumulation_steps += kn_csz // txt_csz
            else:
                gradient_accumulation_steps += math.ceil((kn_csz - (kn_cnt-len(kn))) / txt_csz)
        # print("gradient_accumulation_steps", gradient_accumulation_steps)
        

        kn_cnt = 0
        for kn_csz, txt_csz in chunk_scheme:
            kn_chunk = kn[kn_cnt:kn_cnt+kn_csz]
            txt_chunk = txt[kn_cnt:kn_cnt+kn_csz]
            kn_txt_chunk = kn_txt[kn_cnt:kn_cnt+kn_csz]
            txt_stem_chunk = txt_stem[kn_cnt:kn_cnt+kn_csz]
            kn_cnt += kn_csz
            
            input_ids_kn, attention_mask_kn, _, _ = padding_fn(kn_chunk, pad_id=tokenizer.pad_token_id, gist_token_ids=gist_token_ids)


            # deduplicate the kn_chunk
            input_ids_kn, uni_map = input_ids_kn.unique(dim=0, return_inverse=True, sorted=False)
            anti_map = [-1 for _ in range(len(uni_map))]
            for ii,rr in enumerate(uni_map.tolist()):
                anti_map[rr]=ii
            anti_map = torch.tensor([rr for rr in anti_map if rr!=-1], device=attention_mask_kn.device)
            attention_mask_kn = attention_mask_kn.index_select(dim=0, index=anti_map)


            kn_bsz, _ = input_ids_kn.shape

            for r, (txt_batch, kn_txt_batch, txt_stem_batch) in enumerate(zip(_chunks(txt_chunk, txt_csz), 
                                                                            _chunks(kn_txt_chunk, txt_csz), 
                                                                            _chunks(txt_stem_chunk, txt_csz)
                                                                            )):

                txt_bsz = len(txt_batch)
                input_ids, attention_mask, _, _ = padding_fn(kn_txt_batch, pad_id=tokenizer.pad_token_id, gist_token_ids=gist_token_ids)
                input_ids_query, attention_mask_query, labels, T_labels = padding_fn(txt_batch, bos_token_id=tokenizer.bos_token_id, txt_stem_chunk=txt_stem_batch,
                                                                           pad_id=tokenizer.pad_token_id, gist_token_ids=gist_token_ids)
                
                # print("kn", input_ids_kn.shape, sep="\n")
                # print("input_ids", input_ids.shape, sep="\n")
                # print("input_ids_query", input_ids_query, sep="\n")
                # print(attention_mask, attention_mask.shape)
                # print("labels", labels, labels.shape, sep="\n")

                # txt_bsz = len(input_ids_query)

                gist_pool_idx = torch.zeros(txt_bsz, kn_bsz+1, dtype=torch.int) # plus one to consider the zero gist key and value  
                # gist_pool_idx = torch.zeros(txt_bsz, kn_bsz) 
                col = r*txt_csz
                for i in range(txt_bsz):
                    gist_pool_idx[i,uni_map[col]+1] = 1
                    # gist_pool_idx[i,uni_map[col]] = 1
                    col += 1      
                # print("gist_pool_idx", gist_pool_idx, gist_pool_idx.shape)

                input_ids_ice, attention_mask_ice = mask_gist(input_ids, attention_mask, gist_token_ids=gist_token_ids, pad_id=tokenizer.pad_token_id)

                input_ids_ice, attention_mask_ice, T_labels = (input_ids_ice.to(accelerator.device), 
                                                                attention_mask_ice.to(accelerator.device),
                                                                T_labels.to(accelerator.device))
                input_ids_query, attention_mask_query, input_ids, attention_mask, labels, gist_pool_idx = (input_ids_query.to(accelerator.device),
                                                                                                            attention_mask_query.to(accelerator.device),
                                                                                                            input_ids.to(accelerator.device), 
                                                                                                            attention_mask.to(accelerator.device), 
                                                                                                            labels.to(accelerator.device),
                                                                                                            gist_pool_idx.to(accelerator.device),
                                                                                                            )


                # construct the left and right shift logits mask
                atten_mask = attention_mask.clone()
                atten_mask[reverse_cumsum(input_ids == gist_token_ids) > 0] = 0
                gist_loc = torch.nonzero(input_ids == gist_token_ids)
                l_shift_atten_mask = atten_mask.clone()
                r_border_idx = (gist_loc[:,-1] + atten_mask.cumsum(dim=-1)[:,-1]).to(torch.int)
                for t in range(l_shift_atten_mask.shape[0]):
                    l_shift_atten_mask[t, gist_loc[t,-1].item()+1] = 0 # mask the "\n" token
                    l_shift_atten_mask[t, r_border_idx[t].item()] = 0
                
                # r_shift_atten_mask = atten_mask
                # for t in range(r_shift_atten_mask.shape[0]):
                #     r_shift_atten_mask[t, gist_loc[t,-1]+1] = 0            
                l_shift_atten_mask = l_shift_atten_mask.to(torch.bool)
                # r_shift_atten_mask = r_shift_atten_mask.to(torch.bool)
                # print("l_shift_atten_mask", l_shift_atten_mask)
                # print("r_shift_atten_mask", r_shift_atten_mask)


                atten_mask_q = attention_mask_query.clone()
                if tokenizer.bos_token_id is not None and (input_ids_query[:,0]==tokenizer.bos_token_id).all():
                    atten_mask_q[:,0] = 0 # erase bos token
                    r_shift_atten_mask_q = atten_mask_q.clone()
                    r_shift_atten_mask_q[:,1] = 0
                    l_shift_atten_mask_q = atten_mask_q
                    r_border_idx = atten_mask_q.cumsum(dim=-1)[:,-1].to(torch.int)
                    for t in range(l_shift_atten_mask_q.shape[0]):
                        l_shift_atten_mask_q[t, r_border_idx[t].item()] = 0 
                    query_input_ids = input_ids_query[:,1:]
                    query_attention_mask = attention_mask_query[:,1:]
                    query_labels = labels[:,1:]
                    query_T_labels = T_labels[:,1:]
                else:
                    r_shift_atten_mask_q = atten_mask_q.clone()
                    r_shift_atten_mask_q[:,0] = 0
                    l_shift_atten_mask_q = atten_mask_q
                    r_border_idx = atten_mask_q.cumsum(dim=-1)[:,-1].to(torch.int) - 1
                    for t in range(l_shift_atten_mask_q.shape[0]):
                        l_shift_atten_mask_q[t, r_border_idx[t].item()] = 0       
                    query_input_ids = input_ids_query
                    query_attention_mask = attention_mask_query
                    query_labels = labels     
                    query_T_labels = T_labels       
                l_shift_atten_mask_q = l_shift_atten_mask_q.to(torch.bool)
                r_shift_atten_mask_q = r_shift_atten_mask_q.to(torch.bool)
                # print("l_shift_atten_mask_q", l_shift_atten_mask_q)
                # print("r_shift_atten_mask_q", r_shift_atten_mask_q)


                
                with torch.no_grad():
                    outputs_ice = T_model(input_ids=input_ids_ice, attention_mask=attention_mask_ice)
                    outputs_query = T_model(input_ids=input_ids_query, attention_mask=attention_mask_query)
                    lhs_ice = outputs_ice.last_hidden_state
                    lhs_query = outputs_query.last_hidden_state
                    golden_ice = -fused_CE(_input=lhs_ice[l_shift_atten_mask], 
                                            target=T_labels[r_shift_atten_mask_q], 
                                            lin_weight=T_lm_head_w.weight if "gpt-j" in model_path.lower() else T_lm_head_w,
                                            bias=T_lm_head_w.bias if "gpt-j" in model_path.lower() else None,
                                            )
                    golden_query = -fused_CE(_input=lhs_query[l_shift_atten_mask_q], 
                                            target=T_labels[r_shift_atten_mask_q],
                                            lin_weight=T_lm_head_w.weight if "gpt-j" in model_path.lower() else T_lm_head_w,
                                            bias=T_lm_head_w.bias if "gpt-j" in model_path.lower() else None,
                                            )
                    token_weights = F.relu(golden_ice - golden_query)

                    # pS_token_weights = token_weights.clone()
                    # start_point = 0
                    # token_num = r_shift_atten_mask_q.cumsum(dim=-1)[:,-1]
                    # for t in range(token_num.shape[0]):
                    #     # bsz_slice = pS_token_weights[start_point: start_point+token_num[t].item()]
                    #     bsz_slice = token_weights[start_point: start_point+token_num[t].item()]
                    #     # bsz_slice[:2] = 0
                    #     bsz_slice_topk, indices = bsz_slice.topk(k=4 if len(bsz_slice)>5 else len(bsz_slice), largest=True, sorted=True)
                    #     # bsz_slice_topk, indices = bsz_slice.topk(k=math.ceil(len(bsz_slice)/2), largest=True, sorted=True)
                    #     pS_token_mask = (bsz_slice >= bsz_slice_topk[-1]).to(torch.int)
                    #     bsz_slice *= pS_token_mask | (labels[t][r_shift_atten_mask_q[t]]!=-100)
                    #     start_point += token_num[t].item()
                    
                    token_weights = token_weights * beta

                    # token_weights = token_weights[labels[r_shift_atten_mask_q]!=-100]
                    # pS_token_weights = pS_token_weights[labels[l_shift_atten_mask_q]==-100]
                    # print("pS_token_weights", pS_token_weights, pS_token_weights.shape, sep="\n")

                    # token_weights[labels[r_shift_atten_mask_q]==-100] *= beta_hat
                    # print("token_weights", token_weights, token_weights.shape, sep="\n")



                # Obtain gist token keys and values
                # print("input_ids_kn", input_ids_kn, input_ids_kn.shape)
                base_model(input_ids=input_ids_kn.to(accelerator.device), 
                            attention_mask=attention_mask_kn.to(accelerator.device),
                            gist_pool=gist_pool,
                            gist_token_ids=gist_token_ids,
                            use_cache=False,
                            )
                
                
                # print(gist_pool[0]["keys"].shape, gist_pool[0]["keys"].requires_grad, sep="\n")
                extra_loss = {"sparsity_loss":[], "p0_loss":[], "pS_loss":[]}
                outputs = model(input_ids=query_input_ids, 
                                attention_mask=query_attention_mask,
                                gist_pool=gist_pool, 
                                gist_pool_idx=gist_pool_idx, 
                                token_weights=token_weights,
                                # teacher_input=lhs_ice[l_shift_atten_mask][labels[r_shift_atten_mask_q]!=-100],
                                # teacher_input=lhs_ice[l_shift_atten_mask][(T_labels[r_shift_atten_mask_q]!=-100) & (token_weights!=0)],
                                teacher_input=lhs_ice[l_shift_atten_mask],
                                teacher_weight=T_lm_head_w,
                                gist_token_ids=gist_token_ids,
                                extra_loss=extra_loss,
                                labels=query_labels,
                                T_labels=query_T_labels,
                                use_cache=False,
                                )
                # print("extra_loss", extra_loss)
                 
                loss, ce_loss, wce_loss, klloss = outputs.loss
                # print("loss", outputs.loss)
                # print("pS_loss", torch.stack(extra_loss["pS_loss"]).mean())

                loss = loss / gradient_accumulation_steps
                accelerator.backward(loss)

                for k,v in extra_loss.items():
                    if k not in rd_extra_loss.keys():
                        rd_extra_loss.update({k:[]})
                    if len(v)>0:
                        rd_extra_loss[k].append(torch.stack(v).mean().item())
                    else:
                        rd_extra_loss[k].append(0)
                rd_extra_loss["CrossEntropy_loss"].append(ce_loss.item())
                rd_extra_loss["Weighted_CrossEntropy_loss"].append(wce_loss.item())
                rd_extra_loss["kl_loss"].append(klloss.item())
    
                for _,value in gist_pool.items():       
                    value["keys"] = value["keys"].detach()
                    value["values"] = value["values"].detach()
                    value["keys"] = torch.tensor([])
                    value["values"] = torch.tensor([]) 

                
        updates = (local_step+1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr(it=updates, warmup_iters=warmup_updates, lr_decay_iters=max_updates, min_lr=min_lr, learning_rate=lr)
        optimizer.step()
        optimizer.zero_grad()
        # print('{}, local_step:{}, lr:{}, updates:{}'.format(input_ids.shape, local_step+1, optimizer.param_groups[0]['lr'], (local_step+1))) 

        loss_set = {}
        loss_set.update({"lr":optimizer.param_groups[0]['lr']})
        for k,v in rd_extra_loss.items():
            if len(v)>0:
                loss_k = sum(v)/len(v)
            else:
                loss_k = 0
            loss_set.update({k:loss_k})
            rd_extra_loss[k].clear()
        accelerator.log(loss_set, step=updates)
        # print(loss_set, updates)

        if updates % save_updates==0:
            accelerator.wait_for_everyone()
            model.save_checkpoint(save_dir=checkpoint_dir, tag="checkpoint_{}".format(updates))    
        

    accelerator.wait_for_everyone()
    model.save_checkpoint(save_dir=checkpoint_dir, tag="model")
    accelerator.end_training() 
            




if __name__=="__main__":
    main()

