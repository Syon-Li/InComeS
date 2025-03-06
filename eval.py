import torch
import argparse
import json
import os
import numpy as np
from eval_utils import load_zsre, load_CF, load_dune, load_portability_s_replacement, \
    load_portability_inverse, load_portability_one_hop, load_dune_debiasing, load_mquake, eval_edit_quality
from utils import _chunks
from safetensors.torch import load_file
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
# from zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from edit_attention_llama import Edit_LlamaForCausalLM
# from edit_attention_Ro import Edit_LlamaForCausalLM
# from edit_attention_gptj import Edit_GPTJForCausalLM
# from edit_attention_causal import Edit_LlamaModel, Edit_LlamaForCausalLM
# from edit_attention_gist import Edit_LlamaModel, Edit_LlamaForCausalLM
# from edit_attention_causal_agg import Edit_LlamaModel, Edit_LlamaForCausalLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():

    # model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # model_path = "TinyLlama/TinyLlama_v1.1"
    # model_path = "meta-llama/Llama-2-7b-hf"
    # model_path = "meta-llama/Llama-2-7b-chat-hf"
    # model_path = "EleutherAI/gpt-j-6B"



    parser = argparse.ArgumentParser(description='Model editing baselines.')
    parser.add_argument('--method', type=str, choices=["ICE", "gist", "base"], help='the editing method to use')
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-3B", help='the model to use')
    parser.add_argument('--weight_path', type=str, default="meta-llama/Llama-3.2-3B", help='the trained model weight to use')
    parser.add_argument("--checkpoint_tag", type=str, default=None, help="the checkpoint_tag")
    parser.add_argument("--dataset", type=str, default='counterfact', help="the dataset to use")
    parser.add_argument("--batch_edit", type=int, choices=[0,1], default=0, help="whether to use batch edit")
    parser.add_argument("--run_base", type=int, choices=[0,1], default=0, help="whether to do evaluation for base model")
    parser.add_argument('--bsz', type=int, default=100, help='the editing batch size to use') 
    parser.add_argument('--device', type=int, default=0, help='the device to use')  
    parser.add_argument('--edits', type=int, default=2, help='the hop number of the edit') 
    parser.add_argument('--wrt', type=int, default=0, help='whether to write the result file')  
    args = parser.parse_args()

    device = f"cuda:{args.device}"
    # device = "cpu"

    ds_name = args.dataset
    method = args.method
    run_base = args.run_base
    batch_edit = args.batch_edit
    batch_size = args.bsz
    model_path = args.model
    edits = args.edits
    file_path = args.weight_path
    checkpoint_tag = args.checkpoint_tag
    wrt = args.wrt
    transformers_cache = "./transformers_cache"



    model_name = model_path[model_path.rfind("/"):]

    if ds_name.lower() == "zsre":
        requests = load_zsre("./Editing_data/zsre/zsre_mend_eval.json", num=1000)
    elif ds_name.lower() == "counterfact":
        requests = load_CF("./Editing_data/counterfact/counterfact-edit.json", num=100000)
    elif ds_name.lower() == "dune_arithmetic":
        requests = load_dune("./Editing_data/dune/arithmetic.json", num=10000)
    elif ds_name.lower() == "dune_new_info":
        requests = load_dune("./Editing_data/dune/new_info.json", num=10000)
    elif ds_name.lower() == "dune_scientific":
        requests = load_dune("./Editing_data/dune/scientific.json", num=10000)
    elif ds_name.lower() == "dune_debiasing_i":
        requests = load_dune_debiasing("./Editing_data/dune/debiasing_I.json", num=10000)   
    elif ds_name.lower() == "dune_debiasing_ii":
        requests = load_dune_debiasing("./Editing_data/dune/debiasing_II.json", num=10000)      
    elif ds_name.lower() == "portability_subject_replace":
        requests = load_portability_s_replacement(["./Editing_data/portability/Subject Replace/counterfact_subject_replace.json", 
                                                   "./Editing_data/portability/Subject Replace/zsre_subject_replace.json"])
    elif ds_name.lower() == "portability_one_hop":
        requests = load_portability_one_hop(["./Editing_data/portability/One Hop/counterfact_portability_gpt4.json", 
                                             "./Editing_data/portability/One Hop/zsre_mend_eval_portability_gpt4.json"])
    elif ds_name.lower() == "portability_inverse_relation":
        requests = load_portability_inverse(["./Editing_data/portability/Inverse Relation/zsre_inverse_relation.json"])
    elif ds_name.lower() == "mquake":
        requests = load_mquake(["./Editing_data/MQuAKE/MQuAKE-CF-3k-v2.json"], edits=edits, num=1000000)




    print(requests[0])

    if not batch_edit:
        batch_size = len(requests)
            

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    # model.train()
    # print(model.config.attention_dropout)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # print(list(model.named_parameters()))


    pre_res_all = []
    if run_base:
        # pre_res_all = compute_edit_quality_single(records=requests, model=model, tok=tokenizer, gist_pool=None, method="base")    
        pre_res_all = eval_edit_quality(records=requests, model=model, tok=tokenizer, method="base")   


    if method == "gist":
        model.to("cpu")
        # edit_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", cache_dir="./transformers_cache")
        # edit_tokenizer.pad_token = edit_tokenizer.eos_token
        edit_tokenizer = tokenizer
        num_added_toks = edit_tokenizer.add_special_tokens({"additional_special_tokens": ["<GIST>"]})
        gist_token_ids = edit_tokenizer.additional_special_tokens_ids[-1]


        config = AutoConfig.from_pretrained(model_path)
        if "llama" in model_path.lower():
            edit_model = Edit_LlamaForCausalLM(config=config).to(device)
        else:
            edit_model = Edit_GPTJForCausalLM(config=config).to(device)

        # edit_model = Edit_LlamaForCausalLM.from_pretrained(model_path, cache_dir="./transformers_cache", local_files_only=True).to(device)

        edit_model.resize_token_embeddings(len(edit_tokenizer))
        # state_dict = {}
        # files_and_dirs = os.listdir(file_path)
        # for f in files_and_dirs:
        #     print(f)
        #     path = os.path.join(file_path, f)
        #     if os.path.isfile(path) and "json" not in f:
        #         loaded = load_file(path)
        #         state_dict.update(loaded)
        state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir=file_path, tag=checkpoint_tag)
        # print(state_dict)
        
        if "llama" in model_path.lower():
            removed_value = state_dict.pop("model.norm.weight", None)
            # print(removed_value, removed_value.shape)
            if removed_value is not None:
                norm_weight = model.model.norm.weight.clone()
                norm_weight[:len(removed_value)] = removed_value
                state_dict["model.norm.weight"] = norm_weight
        edit_model.load_state_dict(state_dict, strict=False)
        if "gptj" in edit_model.config.model_type.lower():
            edit_model.to(torch.float16)
        
        for n1, p in edit_model.named_modules():
            # if "offset" in n1:
            print(n1)
    else:
        edit_model = model
        edit_tokenizer = tokenizer





    # post_res_all = []
    # # for records in _chunks(requests, batch_size):
    # if batch_edit:
    #     records, knowledge_set = [], []
    #     for r,record in enumerate(requests):
    #         # print(records, len(records))
    #         # print(type(model), type(edit_model))
            
    #         if isinstance(record["knowledge"], str):
    #             knowledge_set.append(record["knowledge"])
    #         elif isinstance(record["knowledge"], list):
    #             knowledge_set.extend(record["knowledge"])

    #         records.append(record)

    #         if (ds_name.lower()!="mquake" and len(knowledge_set) >= batch_size) or (ds_name.lower()=="mquake" and len(records) == batch_size):
    #             # knowledge_set = list(set(knowledge_set))

    #             # deduplicate and do not change the data order
    #             knowledge_set = list(dict.fromkeys(knowledge_set))

    #         if (ds_name.lower()!="mquake" and len(knowledge_set) == batch_size) or (ds_name.lower()=="mquake" and len(records) == batch_size):
    #             # for _,v in gist_pool.items():
    #             #     v['keys'] = torch.tensor([])
    #             #     v["values"] = torch.tensor([])  
    #             # post_res = compute_edit_quality_batch(records=records, model=edit_model, tok=edit_tokenizer, 
    #             #                                     gist_pool=gist_pool, method=method, knowledge_set=knowledge_set)
    #             post_res = eval_edit_quality(records=records, model=edit_model, tok=edit_tokenizer, 
    #                                         method=method, knowledge_set=knowledge_set, batch=True)
    #             post_res_all.extend(post_res)
    #             knowledge_set.clear()
    #             records.clear()
                

    #     if len(records)>0:
    #         knowledge_set = list(set(knowledge_set))
    #         # for _,v in gist_pool.items():
    #         #     v['keys'] = torch.tensor([])
    #         #     v["values"] = torch.tensor([]) 
    #         post_res = eval_edit_quality(records=records, model=edit_model, tok=edit_tokenizer, 
    #                                     method=method, knowledge_set=knowledge_set, batch=True)
    #         post_res_all.extend(post_res)            
    # else:
    #     # post_res_all = compute_edit_quality_single(records=requests, model=edit_model, tok=edit_tokenizer, 
    #     #                                         gist_pool=gist_pool, method=method,)
    #     post_res_all = eval_edit_quality(records=requests, model=edit_model, tok=edit_tokenizer, 
    #                                     method=method, batch=False)        




    # results = {}
    # if len(pre_res_all) > 0:
    #     rewrite_acc_base, rephrase_acc_base = 0, 0
    #     locality_base = {}
    #     for pre_res in pre_res_all:
    #         # print(post_res, pre_res)
    #         # print(gist_pool, gist_pool[0]["keys"].shape)

    #         for key, value in pre_res["locality_output"].items():
    #             # post_res.update({"locality_acc": np.mean(np.equal(pre_res["locality_output"][key], value)).item()})
    #             if key not in locality_base.keys():
    #                 locality_base.update({key: 0})
    #             # locality_base[key] += pre_res["locality_output"][key][0]
    #             # locality_base[key] += np.mean(np.equal(pre_res["locality_output"][key], value)).item()
    #         rewrite_acc_base += pre_res["rewrite_acc"][0]
    #         rephrase_acc_base += pre_res["rephrase_acc"][0]  

    #     for k,v in locality_base.items():
    #         locality_base[k] = v/len(requests)    
    #     results.update({"Pre-edit":{"rewrite_acc": rewrite_acc_base/len(requests), 
    #                                  "rephrase_acc": rephrase_acc_base/len(requests), 
    #                                  "locality":locality_base}})


    # rewrite_acc, rephrase_acc = 0, 0
    # locality = {}
    # for i, post_res in enumerate(post_res_all):
    #     # print(post_res, pre_res)
    #     # print(gist_pool, gist_pool[0]["keys"].shape)

    #     if len(pre_res_all) > 0:
    #         pre_res = pre_res_all[i]

    #         for key, value in post_res["locality_output"].items():
    #             # post_res.update({"locality_acc": np.mean(np.equal(pre_res["locality_output"][key], value)).item()})
    #             if key not in locality.keys():
    #                 locality.update({key: 0})

    #             locality[key] += np.mean(np.equal(pre_res["locality_output"][key], value)).item()
    #             # locality[key] += value[0]
       
    #     rewrite_acc += post_res["rewrite_acc"][0]
    #     rephrase_acc += post_res["rephrase_acc"][0]
        
    # for k,v in locality.items():
    #     locality[k] = v/len(requests)

    # results.update({"Post-edit":{"rewrite_acc": rewrite_acc/len(requests), "rephrase_acc": rephrase_acc/len(requests), "locality":locality}})



    # run_config = {
    #             "model_id": model_path,
    #             "editing_method": method,
    #             "file_path": file_path,
    #             "batch_size": batch_size,
    #             "batch_edit": batch_edit,
    #             "edits": edits if edits else 1,
    #             "dataset": ds_name,
    #             "requests_size": len(requests),
    #             }
    # print(run_config)
    # print(results)

    # if wrt:
    #     os.makedirs(f"./Experiment_results/{ds_name}", exist_ok=True)
    #     if ds_name.lower() == "mquake":
    #         path = f"./Experiment_results/{ds_name}/{model_name}-{method}-{edits}-{batch_size}.json"
    #     else:
    #         path = f"./Experiment_results/{ds_name}/{model_name}-{method}-{batch_size}.json"
    #     with open(path,"w") as f:
    #         json.dump(run_config, f)
    #         f.write("\n")
    #         json.dump(results, f)




if __name__=="__main__":
    main()