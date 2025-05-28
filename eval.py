import torch
import argparse
import json
import os
import numpy as np
import pickle
from eval_utils import load_zsre, load_CF, load_dune, load_dune_debiasing, load_mquake, eval_edit_quality
from utils import _chunks, replace_subject, replace_what
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from edit_attention_llama import Edit_LlamaForCausalLM
from edit_attention_qwen import Edit_Qwen2ForCausalLM
from edit_attention_gptj import Edit_GPTJForCausalLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    
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



    model_name = model_path[model_path.rfind("/"):]

    if ds_name.lower() == "zsre":
        requests = load_zsre("./Editing_data/zsre/zsre_mend_eval.json", num=100)
    elif ds_name.lower() == "counterfact":
        requests = load_CF("./Editing_data/counterfact/counterfact-edit.json", num=10)
    elif ds_name.lower() == "dune_arithmetic":
        requests = load_dune("./Editing_data/dune/arithmetic.json", num=10000)
    elif ds_name.lower() == "dune_new_info":
        requests = load_dune("./Editing_data/dune/new_info.json", num=10000)
    elif ds_name.lower() == "dune_scientific":
        requests = load_dune("./Editing_data/dune/scientific.json", num=10000)
    elif ds_name.lower() == "dune_debiasing_i":
        requests = load_dune_debiasing("./Editing_data/dune/debiasing_I.json", num=10000)       
    elif ds_name.lower() == "mquake":
        requests = load_mquake(["./Editing_data/MQuAKE/MQuAKE-CF-3k-v2.json"], edits=edits, num=1000000)                    
    elif ds_name.lower() == "wiki-counterfact":
        with open("./KnowEdit/benchmark/wiki_counterfact/test_cf.json") as f:
            requests = json.load(f)
        for record in requests[:]:
            kn = record["prompt"].strip() + " " + record["target_new"].strip() + "."
            record.update({"knowledge": kn})  
    elif ds_name.lower() == "zsre-knowedit":
        with open("./KnowEdit/benchmark/ZsRE/ZsRE-test-all.json") as f:
            requests = json.load(f) 
        for record in requests:
            kn = replace_what(prompt=record["prompt"].strip(), target_new=record["target_new"].strip())
            record.update({"knowledge": kn}) 


    # requests = requests[:10]

    print(requests[0])

    if not batch_edit:
        batch_size = len(requests)  
            

    model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="flash_attention_2", 
                                                 device_map=None, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token


    # print(list(model.named_parameters()))


    if method == "gist":
        model.to("cpu")
        edit_tokenizer = tokenizer
        num_added_toks = edit_tokenizer.add_special_tokens({"additional_special_tokens": ["<GIST>"]})
        gist_token_ids = edit_tokenizer.additional_special_tokens_ids[-1]

        config = AutoConfig.from_pretrained(model_path, attn_implementation="flash_attention_2", 
                                            torch_dtype=torch.bfloat16, device_map="auto")
        if "llama" in model_path.lower():
            edit_model = Edit_LlamaForCausalLM(config=config)
        elif "qwen" in model_path.lower():
            edit_model = Edit_Qwen2ForCausalLM(config=config)
        else:
            edit_model = Edit_GPTJForCausalLM(config=config)

        edit_model.resize_token_embeddings(len(edit_tokenizer))
        state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir=file_path, tag=checkpoint_tag)
        edit_model.load_state_dict(state_dict, strict=True)
        if "gptj" in edit_model.config.model_type.lower():
            edit_model.to(torch.bfloat16)
    else:
        edit_model = model
        edit_tokenizer = tokenizer
    
    edit_model.to(device)
    edit_model.to(torch.bfloat16)


    pre_res_all = []
    if run_base:   
        # model.to(device)
        pre_res_all = eval_edit_quality(records=requests, model=edit_model, tok=edit_tokenizer, method="base") 


    with open('./Llama-3.2-1B_zsre_eval_1_1_target_head_entropy.pkl', 'rb') as f:
        target_entropy = pickle.load(f)


    target_entropy = None

    post_res_all = []
    layer_T = [] if "llama-3.2-1b" in model_path.lower() else None
    if batch_edit:
        records, knowledge_set = [], []
        for r,record in enumerate(requests):
            # print(records, len(records))
            # print(type(model), type(edit_model))
            
            if isinstance(record["knowledge"], str):
                knowledge_set.append(record["knowledge"])
            elif isinstance(record["knowledge"], list):
                knowledge_set.extend(record["knowledge"])
                # knowledge_set.append(record["knowledge"])

            records.append(record)

            if (ds_name.lower()!="mquake" and len(knowledge_set) >= batch_size) or (ds_name.lower()=="mquake" and len(records) == batch_size):
                # knowledge_set = list(set(knowledge_set))

                # deduplicate and do not change the data order
                knowledge_set = list(dict.fromkeys(knowledge_set))

            if (ds_name.lower()!="mquake" and len(knowledge_set) == batch_size) or (ds_name.lower()=="mquake" and len(records) == batch_size):
                post_res = eval_edit_quality(records=records, model=edit_model, tok=edit_tokenizer, target_entropy=target_entropy, record_stats=layer_T,
                                            method=method, knowledge_set=knowledge_set, batch=True)
                post_res_all.extend(post_res)
                knowledge_set.clear()
                records.clear()
                

        if len(records)>0:
            knowledge_set = list(set(knowledge_set))
            post_res = eval_edit_quality(records=records, model=edit_model, tok=edit_tokenizer, target_entropy=target_entropy, record_stats=layer_T,
                                        method=method, knowledge_set=knowledge_set, batch=True)
            post_res_all.extend(post_res)          
    else:
        post_res_all = eval_edit_quality(records=requests, model=edit_model, tok=edit_tokenizer, target_entropy=target_entropy, record_stats=layer_T,
                                        method=method, batch=False)  


    if layer_T is not None and len(layer_T)>0:
        if isinstance(layer_T[0], list):
            rnt = torch.tensor(layer_T).mean(dim=0)
            print(rnt, rnt.shape)


    results = {}
    if len(pre_res_all) > 0:
        rewrite_acc_base, rephrase_acc_base = [], []
        locality_base, portability_base = {}, {}
        for pre_res in pre_res_all:
            # print(post_res, pre_res)
            for key, value in pre_res["portability_output"].items():
                if key not in portability_base.keys():
                    portability_base.update({key: []})
                portability_base[key].extend(value)

            # for key, value in pre_res["locality_output"].items():
            #     if key not in locality_base.keys():
            #         locality_base.update({key: []})
            #     locality_base[key].append(value[0])

            rewrite_acc_base.append(pre_res["rewrite_acc"][0])
            rephrase_acc_base.append(pre_res["rephrase_acc"][0])

        for k,v in portability_base.items():
            portability_base[k] = sum(v)/len(v)
        for k,v in locality_base.items():
            locality_base[k] = sum(v)/len(v)
        results.update({"Pre-edit":{"rewrite_acc": sum(rewrite_acc_base) / len(rewrite_acc_base), 
                                     "rephrase_acc": sum(rephrase_acc_base) / len(rephrase_acc_base), 
                                     "portability":portability_base,
                                     "locality":locality_base,
                                     }})


    rewrite_acc, rephrase_acc = [], []
    portability, locality = {}, {}
    for i, post_res in enumerate(post_res_all):
        # print(post_res, pre_res)
        if len(pre_res_all) > 0:
            pre_res = pre_res_all[i]

            for key, value in post_res["locality_output"].items():
                if key not in locality.keys():
                    locality.update({key: []})
                for ii, value_item in enumerate(value):
                    locality[key].append(np.mean(np.equal(pre_res["locality_output"][key][ii], value_item)).item())

        # for key, value in post_res["locality_output"].items():
        #     if key not in locality.keys():
        #         locality.update({key: []})
        #     locality[key].append(value[0])

        for key, value in post_res["portability_output"].items():
            if key not in portability.keys():
                portability.update({key: []})
            portability[key].extend(value)
       
        rewrite_acc.append(post_res["rewrite_acc"][0])
        rephrase_acc.append(post_res["rephrase_acc"][0])

    # print("pre_res_all", pre_res_all)
    # print("post_res_all", post_res_all)

    for k,v in portability.items():
        portability[k] = sum(v)/len(v)

    for k,v in locality.items():
        locality[k] = sum(v)/len(v)

    results.update({"Post-edit":{"rewrite_acc": sum(rewrite_acc)/len(rewrite_acc), "rephrase_acc": sum(rephrase_acc)/len(rephrase_acc), 
                                 "portability":portability, "locality":locality}})



    run_config = {
                "model_id": model_path,
                "editing_method": method,
                "file_path": file_path,
                "batch_size": batch_size,
                "batch_edit": batch_edit,
                "edits": edits if edits else 1,
                "dataset": ds_name,
                "requests_size": len(requests),
                }
    print(run_config)
    print(results)

    if wrt:
        os.makedirs(f"./Experiment_results/{ds_name}", exist_ok=True)
        if ds_name.lower() == "mquake":
            path = f"./Experiment_results/{ds_name}/{model_name}-{method}-{edits}-{batch_size}.json"
        else:
            path = f"./Experiment_results/{ds_name}/{model_name}-{method}-{batch_size}.json"
        with open(path,"w") as f:
            json.dump(run_config, f)
            f.write("\n")
            json.dump(results, f)




if __name__=="__main__":
    main()