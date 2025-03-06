# Used split_batches for fsdp, but dispatch_batches for deepspeed

from transformers import AutoTokenizer, AutoModelForCausalLM
from edit_attention_llama import Edit_LlamaForCausalLM
from edit_attention_gptj import Edit_GPTJForCausalLM
from edit_attention_qwen import Edit_Qwen2ForCausalLM
# from edit_attention_Ro import Edit_LlamaForCausalLM
# from edit_attention_causal import Edit_LlamaModel, Edit_LlamaForCausalLM
# from edit_attention_causal_agg import Edit_LlamaModel, Edit_LlamaForCausalLM
from utils import wrap_collate_fn, HFIterD, remove_gist, reverse_cumsum, remove_context, \
collate_fn, padding_fn, _chunks, get_lr, set_seed
from accelerate import Accelerator, DataLoaderConfiguration, FullyShardedDataParallelPlugin, DeepSpeedPlugin, init_empty_weights
from accelerate.utils import GradientAccumulationPlugin, get_active_deepspeed_plugin, broadcast_object_list, BnbQuantizationConfig, load_and_quantize_model
from liger_kernel.transformers import apply_liger_kernel_to_llama, LigerFusedLinearCrossEntropyLoss, LigerCrossEntropyLoss, LigerFusedLinearJSD, apply_liger_kernel_to_qwen2
from liger_kernel.transformers.kl_div import LigerKLDIVLoss
from deepspeed.utils import safe_get_full_fp32_param
import torch.nn.functional as F
import argparse
import torch
import random
import copy
import os
import datetime
import wandb




os.environ["NCCL_DEBUG"] = "INFO"

now = datetime.datetime.now()




def main():

    set_seed(42)

    parser = argparse.ArgumentParser(description='Pretraining.')
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-1B", help='the model to use')
    parser.add_argument('--gradaccu', type=int, default=8, help='number of gradient steps to accumulate')
    parser.add_argument('--gist_bsz', type=int, default=128, help='processing gist batch size')
    parser.add_argument("--gist_num", type=int, default=100, help="number of gist token activations to keep")
    args = parser.parse_args()

    start_point = 0
    end_point = 256*8*3

    lr = 1e-5
    min_lr = 1e-6
  
    alpha = 1
    beta = 0.5


    # max_updates = 4000
    # warmup_updates = max_updates // 5
    # save_updates = max_updates // 5

    max_updates = 3500
    warmup_updates = max_updates // 10
    save_updates = max_updates // 5

    # max_updates = 1600
    # warmup_updates = max_updates // 10
    # save_updates = max_updates // 5

    # max_updates = 2000
    # warmup_updates = max_updates // 10
    # save_updates = max_updates // 5

    accu_num = args.gradaccu
    model_path = args.model

    model_name = model_path[model_path.rfind("/")+1:]

    loss_f_name = "train_loss_{}_{}".format(model_name, now.strftime('%Y-%m-%d_%H-%M'))

    
    batch_size = args.gist_bsz

    gist_num = args.gist_num

    # ds_name = "slimpajama_10B"
    # ds_name = "paranmt5m_all-nli"
    # ds_name = "paranmt5m_all-nli-Lamini"
    # ds_name = "paranmt5m_all-nli-Lamini-math-ZSRE-CF"
    # ds_name = "paranmt5m_all-nli-Lamini-ZSRE-CF"
    # ds_name = "nli-MCQ-ZSRE-CF"
    # ds_name = "MCQ-ZSRE-CF-gsm8k"
    ds_name = "wikipara-Lamini-nli-MCQ-ZSRE-CF"
    # ds_name = "paranmt5m_all-nli-Lamini-math"
    # ds_name = "paranmt5m_all-nli-Lamini-math-ZSRE-CF-slimpajama"
    extra_info = "deduplication, use layer 8-15, //10 warm-up"

    checkpoint_dir = "./checkpoints/{}/checkpoints_{}".format(ds_name, now.strftime('%Y-%m-%d_%H-%M'))
    transformers_cache = "./transformers_cache"

    

    if "llama" in model_path.lower():
        apply_liger_kernel_to_llama()
        model = Edit_LlamaForCausalLM.from_pretrained(model_path, local_files_only=True)
    elif "gpt-j" in model_path.lower():       
        model = Edit_GPTJForCausalLM.from_pretrained(model_path, local_files_only=True)
        # model.gradient_checkpointing_enable()
    elif "qwen" in model_path.lower():
        # apply_liger_kernel_to_qwen2()
        model = Edit_Qwen2ForCausalLM.from_pretrained(model_path, local_files_only=True, attn_implementation="sdpa")
    origin_model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, attn_implementation="sdpa")
 

    # print(model.config)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    # origin_model.model.gradient_checkpointing = True

    tokenizer.pad_token = tokenizer.eos_token
    num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens": ["<GIST>"]})
    model.resize_token_embeddings(len(tokenizer))
    # print(tokenizer.pad_token_id, tokenizer.additional_special_tokens_ids)
    gist_token_ids = tokenizer.additional_special_tokens_ids[-1]

    

    with torch.no_grad():
        if "llama" in model_path.lower() or "qwen" in model_path.lower():
            model.model.embed_tokens.weight[-1] = model.model.embed_tokens.weight[:-1].mean(0)
        elif "gpt-j" in model_path.lower():
            model.transformer.wte.weight[-1] = model.transformer.wte.weight[:-1].mean(0)
        model.lm_head.weight[-1] = model.lm_head.weight[:-1].mean(0)


    gist_pool = {}
    for i in range(model.config.num_hidden_layers):
        gist_pool.update({i:{"keys":torch.tensor([]), "values":torch.tensor([])}})
    
        

    wandb.login(key="bd74b6c0b26711d4789414389f4e82b90dbf0bc0")

    if "llama" in model_path.lower():
        deepspeed_plugins = {"student":DeepSpeedPlugin(offload_optimizer_device=None, zero_stage=2), 
                            "teacher":DeepSpeedPlugin(offload_param_device=None, zero_stage=3, zero3_save_16bit_model=True,
                                                    #   hf_ds_config="./zero_stage3_config.json",
                                                    ),}
        accelerator = Accelerator(
                                dataloader_config=DataLoaderConfiguration(dispatch_batches=False,), 
                                deepspeed_plugin=deepspeed_plugins,
                                log_with="wandb",
                                )
        
    elif "gpt-j" in model_path.lower() or "qwen" in model_path.lower():
        # accelerator = Accelerator(
        #                         dataloader_config=DataLoaderConfiguration(dispatch_batches=False,), 
        #                         log_with="wandb",
        #                         )

        deepspeed_plugins = {"student":DeepSpeedPlugin(offload_optimizer_device="cpu", zero_stage=2), 
                            "teacher":DeepSpeedPlugin(offload_param_device="cpu", offload_optimizer_device="cpu", zero_stage=3, zero3_save_16bit_model=True,),}
        accelerator = Accelerator(
                                dataloader_config=DataLoaderConfiguration(dispatch_batches=False,), 
                                deepspeed_plugin=deepspeed_plugins,
                                log_with="wandb",
                                )
        
        # accelerator = Accelerator(
        #                         dataloader_config=DataLoaderConfiguration(dispatch_batches=False,), 
        #                         deepspeed_plugin=DeepSpeedPlugin(offload_optimizer_device="cpu", zero_stage=2), 
        #                         log_with="wandb",
        #                         )

    
    # Broadcast the data from process 0 to all other processes
    broadcast_object_list([checkpoint_dir], from_process=0)
    print(accelerator.device, accelerator.process_index, accelerator.distributed_type, accelerator.num_processes, checkpoint_dir)




    # hf_f_path = ["cestwc/adapted-paranmt5m", "sentence-transformers/all-nli", "MBZUAI/LaMini-instruction", "qwedsacf/competition_math", "openai/gsm8k", "lighteval/MATH-Hard"]
    # subset = [None, "pair", None, None, "main", None]
    # weights = [5.31, 0.328, 2.7, 0.49, 0.03]
    # weights = [5.31, 0.328, 0.49, 0.03]

    hf_f_path = ["ltg/en-wiki-paraphrased", "MBZUAI/LaMini-instruction", "sentence-transformers/all-nli", "allenai/openbookqa", 
                 "allenai/qasc", "openlifescienceai/medmcqa", "NASP/neteval-exam"]
    subset = [None, None, "pair", "main", None, None, None]
    weights = [5, 1.5, 0.328, 0.00596, 0.00813, 0.183, 0.005269, 0.49, 0.03]
    # weights = [4.85, 1.5, 0.328, 0.00596, 0.00813, 0.183, 0.005269, 0.49, 0.03]
    # weights = [5.1, 1.7, 0.5, 0.02, 0.02, 0.3, 0.02, 0.6, 0.05]

    # hf_f_path = ["ltg/en-wiki-paraphrased", "MBZUAI/LaMini-instruction", "sentence-transformers/all-nli", "allenai/openbookqa", 
    #              "allenai/qasc", "openlifescienceai/medmcqa", "NASP/neteval-exam"]
    # subset = [None, None, "pair", "main", None, None, None]
    # # weights = [3, 0.85, 0.328, 0.00596, 0.00813, 0.183, 0.005269, 0.49, 0.03]
    # weights = [3, 0.85, 0.4, 0.01, 0.01, 0.2, 0.01, 0.5, 0.03]

    # hf_f_path = ["allenai/openbookqa", "allenai/qasc", "openlifescienceai/medmcqa", "NASP/neteval-exam", "openai/gsm8k",]
    # subset = ["main", None, None, None, "main"]
    # weights = [0.00596, 0.00813, 0.183, 0.005269, 0.00747, 0.49, 0.03]

    f_path = ["./Editing_data/zsre/zsre_mend_train.json", "./Editing_data/counterfact/counterfact-train.json"]
    
    split = ["train", "train", "train", "train", "train", "train", "train"]


    ds = HFIterD(hf_f_path=hf_f_path, f_path=f_path, subset=subset, 
                 split=split, tokenizer=tokenizer, weights=weights, 
                 lines=end_point)
    train_dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    if "llama" in model_path.lower():
        active_plugin = get_active_deepspeed_plugin(accelerator.state)
        assert active_plugin is deepspeed_plugins["student"], "student deepspeed plugin was not activated"
        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

        accelerator.state.select_deepspeed_plugin("teacher")
        T_model, _ = accelerator.prepare(origin_model, train_dataloader)
        # txt_kn_csz = [(8,8), (16,16), (32,16), (64,8), (100,4)]
        # gist_bsz_w = [0.05, 0.05, 0.15, 0.25, 0.5]
        # gist_bsz_w = [0.05, 0.05, 0.15, 0.15, 0.6]
        # gist_bsz_w = [0.05, 0.15, 0.2, 0.3, 0.3]

        txt_kn_csz = [(32,16), (64,8), (100,8)]
        gist_bsz_w = [0.25, 0.25, 0.5]

        # txt_kn_csz = [(32,16)]
        # gist_bsz_w = [1]   

    elif "gpt-j" in model_path.lower() or "qwen" in model_path.lower():
        # model, T_model, optimizer, train_dataloader = accelerator.prepare(model, origin_model, optimizer, train_dataloader)

        # # fsdp "_is_root" error
        # with torch.no_grad():
        #     model(input_ids=torch.tensor([[1,1,gist_token_ids]], dtype=torch.int, device=model.device), 
        #         attention_mask=torch.tensor([[1,1,1]], dtype=torch.int, device=model.device),
        #         gist_token_ids=gist_token_ids,
        #         gist_pool=copy.deepcopy(gist_pool))
        #     T_model(torch.tensor([[1,1]]), torch.tensor([[1,1]]))



        active_plugin = get_active_deepspeed_plugin(accelerator.state)
        assert active_plugin is deepspeed_plugins["student"], "student deepspeed plugin was not activated"
        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

        # T_lm_head_w = torch.concat([origin_model.lm_head.weight, torch.zeros_like(origin_model.lm_head.weight[-1][None,...])], dim=0)
        T_lm_head_w = origin_model.lm_head.weight
        accelerator.state.select_deepspeed_plugin("teacher")
        T_model, _ = accelerator.prepare(origin_model.model, train_dataloader)


        # model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
        # T_model = origin_model.to(model.device)


        txt_kn_csz = [(16,16)]
        gist_bsz_w = [1]



    
    kl_div = LigerKLDIVLoss(reduction="none")
    fused_CE = LigerFusedLinearCrossEntropyLoss(reduction="none")
    fused_kiv = LigerFusedLinearJSD(jsd_beta=1)
    liger_CE = LigerCrossEntropyLoss(reduction="none")



    config = {"model": model_name,
              "save_updates": save_updates,
              "warmup_updates": warmup_updates,
              "hf_f_path": hf_f_path,
              "sample_weights": weights,
              "batch_size": batch_size,
              "txt_kn_bsz": txt_kn_csz,
              "txt_bsz_weight": gist_bsz_w,
              "accu_num": accu_num,
              "gist_num": gist_num,
              "extra_info": extra_info,
              "beta": beta,
              "alpha": alpha,
              "dataset": ds_name,
              "extra_info": extra_info}
    accelerator.init_trackers("gist-model-editing", config=config)
    
    # print(tokenizer.decode([220])) # space

    # if os.path.exists("./dup_stats.json"):
    #     with open("./dup_stats.json", 'w') as file:
    #         pass


    # print("T_model", T_model)
    # print("model", model)
    model.train()
    T_model.eval()
    # print("model.training", model.module.training)
    # print("T_model.training", T_model.module.training) 
    if "llama" in model_path.lower() or "qwen" in model_path.lower():
        base_model_attr = "model"
    else:
        base_model_attr = "transformer"
    base_model = getattr(model.module, base_model_attr)
    T_lm_head_w = T_lm_head_w.to(model.dtype).to(accelerator.device).detach()
    T_lm_head_w_c = torch.concat([T_lm_head_w, torch.zeros_like(T_lm_head_w[-1][None,...])], dim=0)
    # T_lm_head_w = safe_get_full_fp32_param(T_model.module.lm_head.weight)
    # print("T_lm_head_w", T_lm_head_w, T_lm_head_w.shape, T_lm_head_w.requires_grad)
    rd_extra_loss = {"CrossEntropy_loss": [], "Weighted_CrossEntropy_loss":[], "kl_loss":[]}
    kn_pool = []
    # print(model.dtype)
    for local_step, (kn, txt, kn_txt) in enumerate(train_dataloader):      
        # print(T_model.device)
        # print("kn", len(kn))
        # print("txt", txt)
        # print("kn_txt", kn_txt)

        kn_cnt = 0
        while kn_cnt < len(kn):
            kn_csz, txt_csz = random.choices(txt_kn_csz, weights=gist_bsz_w, k=1)[0]

            kn_chunk = kn[kn_cnt:kn_cnt+kn_csz]
            txt_chunk = txt[kn_cnt:kn_cnt+kn_csz]
            kn_txt_chunk = kn_txt[kn_cnt:kn_cnt+kn_csz]
            kn_cnt += kn_csz

            input_ids_kn, attention_mask_kn, _ = padding_fn(kn_chunk, pad_id=tokenizer.pad_token_id, gist_token_ids=gist_token_ids)

            # deduplicate the kn_chunk
            input_ids_kn, uni_map = input_ids_kn.unique(dim=0, return_inverse=True, sorted=False)
            anti_map = [-1 for _ in range(len(uni_map))]
            for ii,rr in enumerate(uni_map.tolist()):
                anti_map[rr]=ii
            anti_map = torch.tensor([rr for rr in anti_map if rr!=-1])
            attention_mask_kn = attention_mask_kn.index_select(dim=0, index=anti_map)


            # if len(kn_pool)>gist_num:
            #     kn_sample = random.sample(kn_pool, k=gist_num)
            # else:
            #     kn_sample = kn_pool
            # if len(kn_sample)>0:
            #     kn_sample = list(set([tuple(k) for k in kn_sample]) - set([tuple(k) for k in kn_chunk]))
            #     kn_sample = [list(k) for k in kn_sample]
            #     input_ids_kn_s, attention_mask_kn_s, _ = padding_fn(kn_sample, pad_id=tokenizer.pad_token_id, gist_token_ids=gist_token_ids)



            # input_ids_kn, counts = input_ids_kn.unique(dim=0, sorted=False, return_counts=True)
            # uni_counts, occurs = counts.unique(sorted=True, return_counts=True)
            # dup_stats = {str((local_step, kn_csz, len(input_ids_kn))):{}}
            # for uni_co, occ in zip(uni_counts, occurs):
            #     dup_stats[str((local_step, kn_csz, len(input_ids_kn)))].update({uni_co.item(): occ.item()})
            # # print(dup_stats)

            # with open("./dup_stats.json", "a") as file:
            #     json.dump(dup_stats, file)
            #     file.write("\n")
            


            kn_bsz, _ = input_ids_kn.shape

            # increase the txt_csz for better gpu utilization. len(kn_chunk) cannot be changed to kn_bsz, otherwise will cause inconsistency among processes
            # if len(kn_chunk)<kn_csz:
            #     if len(kn_chunk)<=32:
            #         txt_csz = 16
            #     elif len(kn_chunk)<=64:
            #         txt_csz = 8

            for r, (txt_batch, kn_txt_batch) in enumerate(zip(_chunks(txt_chunk, txt_csz), _chunks(kn_txt_chunk, txt_csz))):

                txt_bsz = len(txt_batch)
                input_ids, attention_mask, labels = padding_fn(kn_txt_batch, pad_id=tokenizer.pad_token_id, gist_token_ids=gist_token_ids)
                input_ids_query, attention_mask_query, _ = padding_fn(txt_batch, pad_id=tokenizer.pad_token_id, gist_token_ids=gist_token_ids)
                
                # print(input_ids, input_ids.shape)
                # print(attention_mask, attention_mask.shape)
                # print(labels)


                gist_pool_idx = torch.zeros(txt_bsz, kn_bsz+1) # plus one to consider the zero gist key and value  
                # gist_pool_idx = torch.zeros(txt_bsz, kn_bsz) 
                col = r*txt_csz
                for i in range(txt_bsz):
                    gist_pool_idx[i,uni_map[col]+1] = 1     
                    # gist_pool_idx[i,col+1] = 1
                    col += 1      
                # print("gist_pool_idx", gist_pool_idx, gist_pool_idx.shape)


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
                r_border_idx = (gist_loc[:,-1] + atten_mask.cumsum(dim=-1)[:,-1]).to(torch.long)
                for t in range(l_shift_atten_mask.shape[0]):
                    l_shift_atten_mask[t, r_border_idx[t].item()] = 0
                
                r_shift_atten_mask = atten_mask
                for t in range(r_shift_atten_mask.shape[0]):
                    r_shift_atten_mask[t, gist_loc[t,-1]+1] = 0            
                l_shift_atten_mask = l_shift_atten_mask.to(torch.bool)
                r_shift_atten_mask = r_shift_atten_mask.to(torch.bool)
                # print("l_shift_atten_mask", l_shift_atten_mask)
                # print("r_shift_atten_mask", r_shift_atten_mask)


                atten_mask_q = attention_mask_query.clone()
                atten_mask_q[:,0] = 0 # erase bos token
                r_shift_atten_mask_q = atten_mask_q.clone()
                r_shift_atten_mask_q[:,1] = 0
                l_shift_atten_mask_q = atten_mask_q
                r_border_idx = atten_mask_q.cumsum(dim=-1)[:,-1].to(torch.long)
                for t in range(l_shift_atten_mask_q.shape[0]):
                    l_shift_atten_mask_q[t, r_border_idx[t].item()] = 0 
                l_shift_atten_mask_q = l_shift_atten_mask_q.to(torch.bool)
                r_shift_atten_mask_q = r_shift_atten_mask_q.to(torch.bool)



                input_ids_ice, attention_mask_ice = remove_gist(input_ids, attention_mask, 
                                                                gist_token_ids=gist_token_ids, pad_id=tokenizer.pad_token_id)

                
                with torch.no_grad():
                    outputs_ice = T_model(input_ids=input_ids_ice, attention_mask=attention_mask_ice)
                    outputs_query = T_model(input_ids=input_ids_query, attention_mask=attention_mask_query)
                    lhs_ice = outputs_ice.last_hidden_state
                    lhs_query = outputs_query.last_hidden_state
                    # logits_ice = T_model.lm_head(lhs_ice)
                    golden_ice = -fused_CE(_input=lhs_ice[l_shift_atten_mask], 
                                            target=labels[r_shift_atten_mask], 
                                            lin_weight=T_lm_head_w,
                                            )
                    golden_query = -fused_CE(_input=lhs_query[l_shift_atten_mask_q], 
                                            target=labels[r_shift_atten_mask], 
                                            lin_weight=T_lm_head_w,
                                            )
                    token_weights = F.relu(golden_ice - golden_query) * beta
                    # print("token_weights", token_weights, token_weights.shape)

                    # Augmented gist keys and values
                    # if len(kn_sample)>0:
                    #     _ = model.model(input_ids=input_ids_kn_s.to(accelerator.device), 
                    #                     attention_mask=attention_mask_kn_s.to(accelerator.device),
                    #                     gist_pool=gist_pool,
                    #                     gist_token_ids=gist_token_ids,
                    #                     use_cache=False,
                    #                     )
                
                # gist_pool_idx = torch.concat([torch.zeros(txt_bsz, gist_pool[0]["keys"].shape[0], device=gist_pool_idx.device), gist_pool_idx], dim=-1)


                # Obtain gist token keys and values
                # print("input_ids_kn", input_ids_kn, input_ids_kn.shape)
                base_model(input_ids=input_ids_kn.to(accelerator.device), 
                            attention_mask=attention_mask_kn.to(accelerator.device),
                            gist_pool=gist_pool,
                            gist_token_ids=gist_token_ids,
                            use_cache=False,
                            )
                
                
                # print(gist_pool[14]["keys"].shape, gist_pool[14]["values"].requires_grad)
                # print("input_ids_query", input_ids_query, input_ids_query.shape)
                extra_loss = {"sparsity_loss":[], "p0_loss":[], "pS_loss":[]}
                outputs = base_model(input_ids=input_ids_query[:,1:], 
                                    attention_mask=attention_mask_query[:,1:],
                                    gist_pool=gist_pool, 
                                    gist_pool_idx=gist_pool_idx, 
                                    gist_token_ids=gist_token_ids,
                                    extra_loss=extra_loss,
                                    use_cache=False,
                                    )

                # print("T_lm_head_w", T_lm_head_w.dtype)
                # print("outputs.last_hidden_state", outputs.last_hidden_state.dtype)
                klloss = fused_kiv(student_input=outputs.last_hidden_state[l_shift_atten_mask_q[:,1:]], 
                                   student_weight=model.lm_head.weight, 
                                   teacher_input=lhs_ice[l_shift_atten_mask],
                                   teacher_weight=T_lm_head_w_c,
                                   shift_labels=None,
                                   )
                # print("kl_loss", klloss)
                
                CE_loss = fused_CE(_input=outputs.last_hidden_state[l_shift_atten_mask_q[:,1:]], 
                                    target=labels[r_shift_atten_mask], 
                                    lin_weight=model.lm_head.weight,
                                    bias=model.lm_head.bias if model.lm_head.bias is not None else None,
                                    )



                W_CEloss = (CE_loss * token_weights).mean()
                print("W_CEloss", W_CEloss, W_CEloss.dtype)       
                print("extra_loss", extra_loss)    

                loss = W_CEloss + alpha*klloss
                # loss = CE_loss.mean()
                # print("loss", loss)
                # print("pS_loss", torch.stack(extra_loss["pS_loss"]).mean())

                # del input_ids, attention_mask, labels, input_ids_ice, attention_mask_ice, input_ids_query, attention_mask_query, \
                #     gist_pool_idx, outputs_ice, outputs_query, outputs, l_shift_atten_mask
                # torch.cuda.empty_cache()

                accelerator.backward(loss)

                for k,v in extra_loss.items():
                    if k not in rd_extra_loss.keys():
                        rd_extra_loss[k] = []
                    else:
                        if len(v)>0:
                            rd_extra_loss[k].append(torch.stack(v).mean().item())
                        else:
                            rd_extra_loss[k].append(0)
                rd_extra_loss["CrossEntropy_loss"].append(CE_loss.mean().item())
                # rd_extra_loss["Weighted_CrossEntropy_loss"].append(W_CEloss.item())
                # rd_extra_loss["kl_loss"].append(klloss.item())
    
                for _,value in gist_pool.items():       
                    value["keys"] = value["keys"].detach()
                    value["values"] = value["values"].detach()
                    value["keys"] = torch.tensor([])
                    value["values"] = torch.tensor([]) 
                    # if len(value["keys"]) > gist_num:
                    #     if gist_num != 0:
                    #         value["keys"] = value["keys"][-gist_num:]
                    #         value["values"] = value["values"][-gist_num:]
                    #     else:
                    #         value["keys"] = torch.tensor([])
                    #         value["values"] = torch.tensor([])          
                # print("gist_pool size", value["keys"].shape[0])
    
        # kn_pool.extend(kn)
        # if len(kn_pool)> 1000:
        #     kn_pool = kn_pool[:1000] 
                

        if (local_step+1) % accu_num == 0:
            updates = (local_step+1) // accu_num

            for param_group in optimizer.param_groups:
                param_group['lr'] = get_lr(it=updates, warmup_iters=warmup_updates, lr_decay_iters=max_updates, min_lr=min_lr, learning_rate=lr)
            optimizer.step()
            optimizer.zero_grad()
            # print('{}, local_step:{}, lr:{}, updates:{}'.format(input_ids.shape, local_step+1, optimizer.param_groups[0]['lr'], (local_step+1) // accu_num)) 

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
                # accelerator.save_model(model, "{}/checkpoint_{}".format(checkpoint_dir, updates))  
                model.save_checkpoint(save_dir=checkpoint_dir, tag="checkpoint_{}".format(updates))               
        

    accelerator.wait_for_everyone()
    # accelerator.save_model(model, "{}/model".format(checkpoint_dir))  
    model.save_checkpoint(save_dir=checkpoint_dir, tag="model")
    accelerator.end_training() 
            




if __name__=="__main__":
    main()

