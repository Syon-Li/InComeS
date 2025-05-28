from typing import List, Dict, Optional, Union
import torch
import json
import random
import pickle
import numpy as np
import torch.nn.functional as F
import time
from utils import mask_gist, replace_what, replace_subject



def load_zsre(path:str, num:int=-1):
    with open(path) as f:
        ds = json.load(f)
    requests = []
    for record in ds[:num]:
        if record.get("alt",0)!=0:
            subject = record["subject"]
            prompt = record["src"]
            rephrase_prompt = record["rephrase"]
            target_new = record["alt"]
            ground_truth = record["answers"][0]
            locality = {"neighbors": {"prompt": record["loc"][len("nq question: "):], "ground_truth": record["loc_ans"]}}
            kn = prompt.strip() + ' ' + target_new.strip()
            # kn = replace_what(prompt=prompt, target_new=target_new)
            if subject in prompt:
                requests.append({"prompt":prompt, "target_new":target_new, "ground_truth":ground_truth, "rephrase_prompt":rephrase_prompt, 
                                "locality":locality, "subject":subject, "knowledge":kn})
    
    return requests
                
                

def load_CF(path:str, num:int=-1):
    with open(path) as f:
        ds = json.load(f)
    requests = ds[:num]
    for record in requests:
        record["locality"] = {"neighbors": {"prompt": record["locality_prompt"], "ground_truth": record["locality_ground_truth"]}}
        record["knowledge"] = record["prompt"].strip() + ' ' + record["target_new"].strip() + "."

    return requests



def load_dune(path:str, num:int=-1):
    with open(path) as f:
        ds = json.load(f)
    edits = ds["edits"][:num]
    test_inputs = ds["test_inputs"][:num]

    requests = []
    for edit, inputs in zip(edits, test_inputs):
        if type(inputs) == str:
            # {inputs.strip()} 
            requests.append({"prompt":inputs.strip(), "target_new":edit.strip(), "knowledge": edit.strip()})
        else:
            if type(inputs[0])==list:
                for txt in inputs:
                    lines = txt[0].splitlines()
                    for line in lines:
                        if len(line)>1:
                            if line[0] == txt[1]:
                                target_new = ""
                                for i,word in enumerate(line.split()):
                                    # if i!=0:
                                    target_new += " " + word
                                # requests.append({"prompt":lines[0].strip(), "target_new":target_new.strip(), "knowledge": edit.strip()})
                                requests.append({"prompt":txt[0].strip(), "target_new":target_new.strip(), "knowledge": edit.strip()})
                                break
            else:
                lines = inputs[0].splitlines()
                for line in lines:
                    if line[0] == inputs[1]:
                        # target_new = inputs[1]
                        target_new = ""
                        for i,word in enumerate(line.split()):
                            # if i!=0:
                            target_new += " " + word
                        # requests.append({"prompt":lines[0].strip(), "target_new":target_new.strip(), "knowledge": edit.strip()})
                        requests.append({"prompt":inputs[0].strip(), "target_new":target_new.strip(), "knowledge": edit.strip()})
                        break
    
    return requests


def load_dune_debiasing(path:str, num:int=-1):
    with open(path) as f:
        ds = json.load(f)
    if "edits:" in ds.keys():
        edits = ds["edits:"][:num]
    else:
        edits = ds["edits"][:num]
    test_inputs = ds["test_inputs"][:num]

    requests = []
    for edit, inputs in zip(edits, test_inputs):
        for txt in inputs:
            requests.append({"prompt":txt[0].strip(), "target_new":txt[1].strip(), "knowledge": edit})
    
    return requests




def load_portability_s_replacement(f_path):
    requests = []
    for path in f_path:
        with open(path) as f:
            ds = json.load(f)
        if "counterfact" in path:
            for record in ds[:]:
                subject = record["requested_rewrite"]["subject"]
                al_subject = record["alternative_subject"]
                prompt = record["requested_rewrite"]["prompt"].format(al_subject)
                old_prompt = record["requested_rewrite"]["prompt"].format(subject)
                target_new = record["requested_rewrite"]["target_new"]["str"]
                ground_truth = record["requested_rewrite"]["target_true"]["str"]
                requests.append({"prompt":prompt, "target_new":target_new, "ground_truth":ground_truth, 
                                 "subject":subject, "knowledge":old_prompt.strip() + ' ' + target_new.strip()})
        else:
            for record in ds[:]:
                subject = record["subject"]
                prompt = record["alter_subject_question"]
                old_prompt = record["src"]
                target_new = record["alt"]
                ground_truth = record["answers"][0]
                requests.append({"prompt":prompt, "target_new":target_new, "ground_truth":ground_truth, 
                                 "subject":subject, "knowledge":old_prompt.strip() + ' ' + target_new.strip()})

    return requests



def load_portability_inverse(f_path):
    requests = []
    for path in f_path:
        with open(path) as f:
            ds = json.load(f)

        for record in ds[:]:
            subject = record["subject"]
            prompt = record["inverse question"]
            old_prompt = record["src"]
            target_new = record["alt"]
            ground_truth = record["pred"]

            requests.append({"prompt":prompt, "target_new":subject, "ground_truth":ground_truth, 
                             "subject":subject, "knowledge":old_prompt.strip() + ' ' + target_new.strip()})
    
    return requests




def load_portability_one_hop(f_path):
    requests = []
    for path in f_path:
        with open(path) as f:
            ds = json.load(f)
        if "counterfact" in path:
            for record in ds[:]:
                subject = record["requested_rewrite"]["subject"]
                prompt = record["portability"]["New Question"]
                target_new = record["portability"]["New Answer"]
                old_prompt = record["requested_rewrite"]["prompt"].format(subject)
                old_target_new = record["requested_rewrite"]["target_new"]["str"]
                ground_truth = record["requested_rewrite"]["target_true"]["str"]
                requests.append({"prompt":prompt, "target_new":target_new, "ground_truth":ground_truth, 
                                 "subject":subject, "knowledge":old_prompt.strip() + ' ' + old_target_new.strip()})
        else:
            for record in ds[:]:
                subject = record["subject"]
                prompt = record["portability"]["New Question"]
                target_new = record["portability"]["New Answer"]
                old_prompt = record["src"]
                old_target_new = record["alt"]
                ground_truth = record["pred"]
                requests.append({"prompt":prompt, "target_new":target_new, "ground_truth":ground_truth, 
                                 "subject":subject, "knowledge":old_prompt.strip() + ' ' + old_target_new.strip()})

    return requests




def load_mquake(f_path, edits:int, num:int=-1):
    requests = []
    for path in f_path:
        with open(path) as f:
            ds = json.load(f)    
        for record in ds[:num]:
            kns = []
            if len(record["requested_rewrite"]) == edits:
                for rew in record["requested_rewrite"]:
                    # subjects.append(rew["subject"])
                    # target_new.append(rew["target_new"]["str"])
                    # ground_truth.append(rew["target_true"]["str"])
                    kns.append(rew["prompt"].format(rew["subject"]) + " " + rew["target_new"]["str"] + ".")
                
                new_answer = [record["new_answer"]]
                new_answer.extend(record["new_answer_alias"])
                
                answer = [record["answer"]]
                answer.extend(record["answer_alias"])  

                requests.append({"prompt":record["questions"], "target_new": new_answer, "ground_truth": answer, "knowledge": kns})
                # requests[-1].update({"rephrase_prompt":record["questions"], "rephrase_target_new":record["new_answer"]})
    
    return requests




def load_eval_ds(ds_name, num_lines):
    prompts, ground_truth, target_new, subjects = [], [], [], []
    rephrase_prompts = []
    locality_inputs, portability_inputs = {}, {}
    if ds_name == "counterfact-edit":
        with open("./Editing_data/counterfact/counterfact-edit.json") as f:
            ds = json.load(f)
        requests = ds[:num_lines]
        locality_inputs = {"neighbors":{"prompt":[], "ground_truth":[]}}
        for record in requests:
            prompts.append(record["prompt"])
            ground_truth.append(record["ground_truth"])
            target_new.append(record["target_new"])
            rephrase_prompts.append(record["rephrase_prompt"])
            subjects.append(record["subject"])
            for key in locality_inputs.keys():
                locality_inputs[key]["prompt"].append(record["locality_prompt"])
                locality_inputs[key]["ground_truth"].append(record["locality_ground_truth"])
                
    elif ds_name == "zsre_mend_eval":
        with open("./Editing_data/zsre/zsre_mend_eval.json") as f:
            ds = json.load(f)
        requests = ds[:num_lines]
        locality_inputs = {"neighbors":{"prompt":[], "ground_truth":[]}}
        for record in requests:
            if len(record["alt"])>0:
                prompts.append(record["src"])
                ground_truth.append(record["answers"][0])
                target_new.append(record["alt"])
                rephrase_prompts.append(record["rephrase"])
                subjects.append(record["subject"])
                for key in locality_inputs.keys():
                    locality_inputs[key]["prompt"].append(record["loc"][len("nq question: "):])
                    locality_inputs[key]["ground_truth"].append(record["loc_ans"])     
    
    elif ds_name == "portability_inverse_relation":
        portability_inputs = {"inversion relation":{"prompt":[], "ground_truth":[]}}
        with open("./Editing_data/portability/Inverse Relation/zsre_inverse_relation.json") as f:
            ds = json.load(f)
        for record in ds[:num_lines]:
            if record["subject"] in record["src"]:
                prompts.append(record["src"])
                ground_truth.append(record["answers"][0])
                target_new.append(record["alt"])
                subjects.append(record["subject"])
                portability_inputs["inversion relation"]["prompt"].append(record["inverse question"])
                portability_inputs["inversion relation"]["ground_truth"].append(record["subject"])

    elif ds_name == "portability_one_hop":
        portability_inputs = {"one hop":{"prompt":[], "ground_truth":[]}}
        f_path = ["./Editing_data/portability/One Hop/counterfact_portability_gpt4.json", 
                  "./Editing_data/portability/One Hop/zsre_mend_eval_portability_gpt4.json"]
        for path in f_path:
            with open(path) as f:
                ds = json.load(f)
            if "counterfact" in path:
                for record in ds[:num_lines]:
                    subject = record["requested_rewrite"]["subject"]
                    subjects.append(subject)
                    portability_inputs["one hop"]["prompt"].append(record["portability"]["New Question"])
                    portability_inputs["one hop"]["ground_truth"].append(record["portability"]["New Answer"])
                    prompts.append(record["requested_rewrite"]["prompt"].format(subject))
                    target_new.append(record["requested_rewrite"]["target_new"]["str"])
                    ground_truth.append(record["requested_rewrite"]["target_true"]["str"])        
            elif "zsre" in path:
                for record in ds[:num_lines]:
                    if record["subject"] in record["src"]:
                        subjects.append(record["subject"])
                        portability_inputs["one hop"]["prompt"].append(record["portability"]["New Question"])
                        portability_inputs["one hop"]["ground_truth"].append(record["portability"]["New Answer"])
                        prompts.append(record["src"])
                        target_new.append(record["alt"])
                        ground_truth.append(record["answers"][0])

    elif ds_name == "portability_subject_replace":
        portability_inputs = {"subject replacement":{"prompt":[], "ground_truth":[]}}
        f_path = ["./Editing_data/portability/Subject Replace/counterfact_subject_replace.json", 
                  "./Editing_data/portability/Subject Replace/zsre_subject_replace.json"]
        for path in f_path:
            with open(path) as f:
                ds = json.load(f)
            if "counterfact" in path:
                for record in ds[:num_lines]:
                    subject = record["requested_rewrite"]["subject"]
                    al_subject = record["alternative_subject"]
                    subjects.append(subject)
                    portability_inputs["subject replacement"]["prompt"].append(record["requested_rewrite"]["prompt"].format(al_subject))
                    portability_inputs["subject replacement"]["ground_truth"].append(record["requested_rewrite"]["target_new"]["str"])
                    prompts.append(record["requested_rewrite"]["prompt"].format(subject))
                    target_new.append(record["requested_rewrite"]["target_new"]["str"])
                    ground_truth.append(record["requested_rewrite"]["target_true"]["str"])
            else:
                for record in ds[:num_lines]:
                    if record["subject"] in record["src"]:
                        subjects.append(record["subject"])
                        portability_inputs["subject replacement"]["prompt"].append(record["alter_subject_question"])
                        portability_inputs["subject replacement"]["ground_truth"].append(record["alt"])
                        prompts.append(record["src"])
                        target_new.append(record["alt"])
                        ground_truth.append(record["answers"][0])
    
    elif ds_name.lower() == "wiki-counterfact" or ds_name.lower() == "zsre-knowedit":
        if ds_name.lower() == "wiki-counterfact":
            with open("./KnowEdit/benchmark/wiki_counterfact/test_cf.json") as f:
                ds = json.load(f) 
        else:
            with open("./KnowEdit/benchmark/ZsRE/ZsRE-test-all.json") as f:
                ds = json.load(f)             
        
        portability_inputs.update({"Subject_Aliasing":{"prompt":[], "ground_truth":[]},
                                  "Reasoning":{"prompt":[], "ground_truth":[]},
                                   "Logical_Generalization":{"prompt":[], "ground_truth":[]},})
        locality_inputs.update({"Relation_Specificity":{"prompt":[], "ground_truth":[]},
                                   "Forgetfulness":{"prompt":[], "ground_truth":[]},})
        for record in ds[:num_lines]:
            prompts.append(record["prompt"])
            rephrase_prompts.append(record["rephrase" if ds_name.lower() == "wiki-counterfact" else "rephrase_prompt"])
            subjects.append(record["subject"])
            target_new.append(record["target_new"])
            ground_truth.append(record["ground_truth"])
            for k,v in portability_inputs.items():
                if k in record["portability"].keys():
                    temp_prompts = []
                    temp_answers = []
                    for value in record["portability"][k]:
                        temp_prompts.append(value["prompt"])
                        g_truth = value["ground_truth"]
                        while isinstance(g_truth,list):
                            g_truth = g_truth[0]
                        temp_answers.append(g_truth)
                    portability_inputs[k]["prompt"].append(temp_prompts)
                    portability_inputs[k]["ground_truth"].append(temp_answers)
                else:
                    portability_inputs[k]["prompt"].append(None)
                    portability_inputs[k]["ground_truth"].append(None)                    

            for k,v in locality_inputs.items():
                if k in record["locality"].keys():
                    temp_prompts = []
                    temp_answers = []
                    for value in record["locality"][k]:
                        temp_prompts.append(value["prompt"])
                        g_truth = value["ground_truth"]
                        while isinstance(g_truth,list):
                            g_truth = g_truth[0]
                        temp_answers.append(g_truth)
                    locality_inputs[k]["prompt"].append(temp_prompts)  
                    locality_inputs[k]["ground_truth"].append(temp_answers)  
                else:
                    locality_inputs[k]["prompt"].append(None)  
                    locality_inputs[k]["ground_truth"].append(None)                             
    
    return prompts, rephrase_prompts, ground_truth, target_new, subjects, locality_inputs, portability_inputs
            








def test_prediction_acc_gist(model, tok, knowledge:str, prompts:str, 
                             targets:str, gist_pool: Dict, method:str, head_temperature: torch.Tensor,
                             locality=False, gist_pool_idx=None, loss_record={}):
    # config = model.config if "num_hidden_layers" in model.config.keys() else model.module.config
    config = model.config

    if isinstance(prompts, str):
        knowledge,prompts,targets = [knowledge,], [prompts,], [targets,]

    if method=="ICE":
        k_prompt_target = [k + " " + prompt + " " + target for k, prompt, target in zip(knowledge,prompts,targets)]
        k_prompt = [k + " " + prompt for k, prompt in zip(knowledge,prompts)] 
    else:
        k_prompt_target = [prompt.strip() + " " + target.strip() for prompt, target in zip(prompts,targets)]
        k_prompt = [prompt.strip() for prompt in prompts]
        # k_prompt_target = [instruction + " " + prompt.strip() + " " + target.strip() for prompt, target in zip(prompts,targets)]
        # k_prompt = [instruction + " " + prompt for prompt in prompts] 



    max_prompt_len = max([len(tok.encode(_)) for _ in k_prompt_target]) + 1
    k_prompt_target_tok = tok(
        k_prompt_target,
        padding=True,
        truncation=True,
        max_length=max_prompt_len,
        return_tensors="pt",
    ).to(model.device)
    k_prompt_tok = tok(
        k_prompt,
        padding=True,
        truncation=True,
        max_length=max_prompt_len,
        return_tensors="pt",
    )
    # print(model.config.max_position_embeddings if "llama" in model.config.model_type.lower() else model.config.n_positions)
    assert len(k_prompt_target_tok[0]) <= config.max_position_embeddings if "llama" in config.model_type.lower() or "qwen" in config.model_type.lower() else config.n_positions, "inputs exceed maximum allowed length"

    # print("k_prompt_tok", k_prompt_tok['input_ids'], sep="\n")
    # print("k_prompt_target_tok", k_prompt_target_tok['input_ids'], sep="\n")
    # if gist_pool is not None and gist_pool[0]["keys"].shape[0] > 1:
    if method=="gist" and tok.bos_token_id is not None and (k_prompt_tok['input_ids'][:,0]==tok.bos_token_id).all():
        k_prompt_target_tok['input_ids'] = k_prompt_target_tok['input_ids'][:,1:]
        k_prompt_target_tok['attention_mask'] = k_prompt_target_tok['attention_mask'][:,1:]
        k_prompt_tok['input_ids'] = k_prompt_tok['input_ids'][:,1:]
        k_prompt_tok['attention_mask'] = k_prompt_tok['attention_mask'][:,1:]

    num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in k_prompt_tok['input_ids']]
    num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in k_prompt_target_tok['input_ids'].cpu()]
    prompt_len = [x+y for x,y in zip(num_pad_toks,num_prompt_toks)]


    model.eval()
    with torch.no_grad():
        if method=="gist":
            outputs= model(**k_prompt_target_tok, 
                        gist_token_ids=tok.additional_special_tokens_ids[-1],
                        gist_pool=gist_pool,
                        gist_pool_idx=gist_pool_idx.to(model.device) if gist_pool_idx is not None else None,
                        extra_loss=loss_record,
                        head_temperature=head_temperature,
                        use_cache=False,)   
            # print(gist_pool[0]["keys"].shape)   
                       
        else:
            outputs = model(**k_prompt_target_tok)


        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits
        answers = torch.argmax(logits, dim=-1).squeeze().detach().cpu().numpy().tolist()
        labels = k_prompt_target_tok['input_ids'].squeeze().detach().cpu().numpy().tolist()
        answers = slice_list(answers,prompt_len,left=True)
        labels = slice_list(labels,prompt_len,left=False)
        # print(answers, labels)
        if locality:
            return answers if type(answers[0]) is list else [answers,]
        if isinstance(answers[0], list):
            res = []
            for ans,label in zip(answers,labels):
                temp_acc = np.mean(np.equal(ans, label))
                if np.isnan(temp_acc):
                    continue
                res.append(temp_acc)
            return res
        else:
            return [np.mean(np.equal(answers, labels))]
        



def compute_edit_quality_c(record, model, tok, gist_pool, knowledge_set: List, method:str, head_temperature:torch.Tensor, gist_pool_idx=None, loss_record={}):
    
    knowledge = ""
    # if input_mode=="ICE" or input_mode=="gist":
    if method=="ICE":
        knowledge_set = knowledge_set if knowledge_set else []
        random.shuffle(knowledge_set)
        for k in knowledge_set:
            knowledge += " " + k

    res = {}

    # print(record)
    # print(knowledge)
    if isinstance(record["prompt"], list) and isinstance(record["target_new"], list):
        r_acc = []
        for prompt in record["prompt"]:
            for tar_new in record["target_new"]:
                rewrite_acc = test_prediction_acc_gist(model=model, 
                                                tok=tok, 
                                                knowledge=knowledge, 
                                                prompts=prompt, 
                                                targets=tar_new,
                                                gist_pool=gist_pool,
                                                loss_record=loss_record,
                                                method=method,
                                                head_temperature=head_temperature,
                                                gist_pool_idx=gist_pool_idx,
                )
                r_acc.append(rewrite_acc)
        rewrite_acc = max(r_acc)
    elif isinstance(record["prompt"], str) and isinstance(record["target_new"], str):
        rewrite_acc = test_prediction_acc_gist(model=model, 
                                            tok=tok, 
                                            knowledge=knowledge, 
                                            prompts=record["prompt"], 
                                            targets=record["target_new"],
                                            gist_pool=gist_pool,
                                            loss_record=loss_record,
                                            method=method,
                                            head_temperature=head_temperature,
                                            gist_pool_idx=gist_pool_idx,
        )

    rephrase_acc = None
    if "rephrase_prompt" in record.keys() or "rephrase" in record.keys():
        rephrase_acc = test_prediction_acc_gist(model=model, 
                                                tok=tok, 
                                                knowledge=knowledge, 
                                                prompts=record["rephrase_prompt"] if "rephrase_prompt" in record.keys() else record["rephrase"], 
                                                targets=record["target_new"],
                                                gist_pool=gist_pool, 
                                                loss_record=loss_record,
                                                method=method,
                                                head_temperature=head_temperature,
                                                )

    portability_output = {}
    if "portability" in record.keys():
        for key, value in record["portability"].items():
            if isinstance(value, list):
                port_out_all = []
                for v in value:
                    g_truth = v["ground_truth"]
                    while isinstance(g_truth,list):
                        g_truth = g_truth[0]
                    port_out = test_prediction_acc_gist(model=model, 
                                                        tok=tok, 
                                                        knowledge=knowledge, 
                                                        prompts=v["prompt"], 
                                                        targets=g_truth,
                                                        gist_pool=gist_pool, 
                                                        loss_record=loss_record,
                                                        method=method,
                                                        head_temperature=head_temperature,
                                                        )
                    # print("port_out", port_out)
                    port_out_all.extend(port_out)
                # port_out_all = [sum(port_out_all) / len(value)]
            
            else:
                port_out_all = test_prediction_acc_gist(model=model, 
                                                    tok=tok, 
                                                    knowledge=knowledge, 
                                                    prompts=value["prompt"], 
                                                    targets=value["ground_truth"],
                                                    gist_pool=gist_pool, 
                                                    loss_record=loss_record,
                                                    method=method,
                                                    head_temperature=head_temperature,
                                                    )
            portability_output[key] = port_out_all

    locality_output = {}
    if "locality" in record.keys():
        for key, value in record["locality"].items():
            if isinstance(value, list):
                loc_out_all = []
                for v in value:
                    g_truth = v["ground_truth"]
                    while isinstance(g_truth,list):
                        g_truth = g_truth[0]
                    loc_out = test_prediction_acc_gist(model=model, 
                                                        tok=tok, 
                                                        knowledge=knowledge, 
                                                        prompts=v["prompt"], 
                                                        targets=g_truth,
                                                        gist_pool=gist_pool, 
                                                        loss_record=loss_record,
                                                        locality=True,
                                                        method=method,
                                                        head_temperature=head_temperature,
                                                        )
                    # print("loc_out", loc_out)
                    loc_out_all.extend(loc_out)
                # if isinstance(loc_out_all[0], float):
                #     loc_out_all = [sum(loc_out_all) / len(value)]
            else:
                loc_out_all = test_prediction_acc_gist(model=model, 
                                                    tok=tok, 
                                                    knowledge=knowledge, 
                                                    prompts=value["prompt"], 
                                                    targets=value["ground_truth"],
                                                    gist_pool=gist_pool, 
                                                    loss_record=loss_record,
                                                    locality=True,
                                                    method=method,
                                                    head_temperature=head_temperature,
                                                    )
            locality_output[key] = loc_out_all
    
    res.update({"rewrite_acc": rewrite_acc, "rephrase_acc": rephrase_acc if rephrase_acc else [0], 
                "portability_output": portability_output, "locality_output": locality_output})
    # print("res", res)
    
    return res




def eval_edit_quality(records, model, tok, method:str, target_entropy: torch.Tensor = None, knowledge_set:List = [], record_stats:List = None, batch=False):
    # print("model_config", model.module.config)
    # config = model.config if "num_hidden_layers" in model.config.keys() else model.module.config
    config = model.config

    gist_pool = {}
    for i in range(config.num_hidden_layers):
        gist_pool.update({i:{"keys":torch.tensor([]), "values":torch.tensor([])}})

    res_all = []
    if target_entropy is None:
        head_temperature = None
    if batch and method == "gist":
        contexts = []
        for knowledge in knowledge_set:
            contexts.append(knowledge + f" {tok.additional_special_tokens[-1]}")
        inputs = tok(contexts, padding=True, return_tensors="pt",).to(model.device)

        model.eval()
        extra_loss = {}
        with torch.no_grad():
            start_t = time.time()
            model(**inputs, 
                gist_token_ids=tok.additional_special_tokens_ids[-1],
                gist_pool=gist_pool,
                use_cache=False,)
            end_t = time.time()
            print("Gist editing Time:", end_t-start_t)
            
            if target_entropy is not None or record_stats is not None:
                input_ids, attention_mask = mask_gist(inputs["input_ids"], inputs["attention_mask"], gist_token_ids=tok.additional_special_tokens_ids[-1], pad_id=tok.pad_token_id)
                model(input_ids=input_ids, 
                    attention_mask=attention_mask,
                    gist_token_ids=tok.additional_special_tokens_ids[-1],
                    gist_pool=gist_pool,
                    extra_loss=extra_loss,
                    use_cache=False,)

                if record_stats is not None:
                    extra_loss.clear()
                    for i, kn in enumerate(knowledge_set):
                        gist_pool_idx = torch.zeros(1, len(knowledge_set)+1, device=model.device) # plus one to consider the zero gist key and value 
                        gist_pool_idx[0, i+1] = 1
                        inputs = tok(kn, return_tensors="pt",).to(model.device)
                        model(**inputs,
                            gist_token_ids=tok.additional_special_tokens_ids[-1],
                            gist_pool=gist_pool,
                            extra_loss=extra_loss,
                            use_cache=False,)                        
                        # layer_p0 = []
                        # for layer_idx, gist_w in extra_loss["gist_w"]:
                        #     layer_p0.append(gist_w[...,0].mean().item())
                        # record_stats.append(layer_p0)

                        layer_p0 = []
                        for layer_idx, gist_w in extra_loss["Entropy"]:
                            # print(gist_w.shape)
                            # tem = gist_w[...,0].mean(dim=1)
                            # tem = (gist_w * gist_pool_idx[:,None,None,:]).sum(dim=-1).mean(dim=1)
                            tem = gist_w.mean(dim=1)
                            layer_p0.append(tem)
                        rnt = torch.concat(layer_p0).mean(dim=0)
                        record_stats.append(rnt)

                        extra_loss.clear()
                        # print(kn)
                    # print("record_stats", record_stats)

                # layer_statics = []
                # for layer_idx, gist_w in extra_loss["gist_w"]:
                    
                #     layer_statics.append(gist_w[...,0].mean(dim=(0,1)))
                # record_stats.append(layer_statics)
                # print("record_stats", record_stats)
                
            if target_entropy is not None:
                input_logits = torch.tensor([], device=model.device)
                for layer_idx, gist_logits in extra_loss["gist_logits"]:
                    input_logits = torch.concat([input_logits, gist_logits[None,...]], dim=0)
                head_temperature = get_temperature(input_logits=input_logits, target_entropy=target_entropy.to(model.device), attention_mask=attention_mask)
                # print("head_temperature", head_temperature.mean(dim=-1))
                record_stats.append(head_temperature.mean(dim=-1).tolist())

        print(gist_pool[config.num_hidden_layers-1]["keys"].shape)
    
    if batch and method == "ICE":
        print("number of context", len(knowledge_set))
        knowledge = ""
        for k in knowledge_set:
            knowledge += " " + k
        # print("knowledge", knowledge)
        inputs = tok(knowledge, return_tensors="pt").to(model.device)
        start_t = time.time()
        model(**inputs)
        end_t = time.time()
        print("ICE editing time", end_t-start_t)

    head_entropy_r = torch.zeros(model.config.num_hidden_layers//2, model.config.num_attention_heads, device=model.device)
    for i,record in enumerate(records):
        # if batch:
        #     gist_pool_idx = torch.zeros(1, len(knowledge_set)+1) # plus one to consider the zero gist key and value 
        #     gist_pool_idx[0,i+1] = 1
        # else:
        #     gist_pool_idx = torch.zeros(1, 2) # plus one to consider the zero gist key and value 
        #     gist_pool_idx[0,1] = 1            
            
        gist_pool_idx = None
        if not batch and method!="base":
            knowledge_set = record["knowledge"] if isinstance(record["knowledge"], list) else [record["knowledge"]]
            if method == "gist":
                for _,v in gist_pool.items():
                    v['keys'] = torch.tensor([])
                    v["values"] = torch.tensor([]) 

                contexts = [kn + f" {tok.additional_special_tokens[-1]}" for kn in knowledge_set]
                inputs = tok(contexts, padding=True, return_tensors="pt",).to(model.device)
                with torch.no_grad():
                    _ = model(**inputs, 
                            gist_token_ids=tok.additional_special_tokens_ids[-1],
                            gist_pool=gist_pool,
                            use_cache=False,)
                    if i<5:
                        # print(contexts)
                        print(gist_pool[config.num_hidden_layers-1]["keys"].shape)
            
            if method == "ICE" and i<5:
                print("number of context", len(knowledge_set))
            
        loss_record = {"Entropy":[]}
        res = compute_edit_quality_c(record=record, model=model, tok=tok, 
                                       gist_pool=gist_pool if gist_pool else None, knowledge_set=knowledge_set, method=method, head_temperature=head_temperature,
                                       gist_pool_idx=gist_pool_idx if gist_pool_idx is not None else None, loss_record=loss_record)
        res_all.append(res)

        # print("Entropy:", loss_record["Entropy"])
        # print("topk_gist_w:", loss_record["topk_gist_w"])
        # gist_w = torch.tensor([], device=model.device)
        # for laye_idx, topk_gist_w in loss_record["topk_gist_w"]:
        #     gist_w = torch.concat([gist_w, topk_gist_w[None,...]])
        # gist_w, indices = gist_w.sort(descending=True, dim=-1)
        # print("gist_w", gist_w, sep="\n")
        # print("indices", indices, sep="\n")
        # uni_sort_gist_w, _ = gist_w.reshape(-1).sort(descending=True)
        # print("uni_sort_gist_w", uni_sort_gist_w)

    # head_entropy_r /= len(records)
    # print("head_entropy_r", head_entropy_r)
    # with open(f'./Llama-3.2-1B_zsre_eval_1_1_target_head_entropy.pkl', 'wb') as f:
    #     pickle.dump(head_entropy_r.cpu(), f)
    
    return res_all



def slice_list(matrix,start_indices,left):
    if isinstance(matrix[0], list):
        if left:
            return [row[start_index-1:-1] for row, start_index in zip(matrix, start_indices)]
        else:
            return [row[start_index:] for row, start_index in zip(matrix, start_indices)]
    else:
        if left:
            return matrix[start_indices[0]-1:-1]
        else:
            return matrix[start_indices[0]:]




# input_logits - [layer, bsz, num_heads, q_len, num_gist]
# target_entropy - [layer, heads]
# attention_mask - [bsz, q_len]
def get_temperature(input_logits, target_entropy, attention_mask, lr=1, n_iterations=200):
    eps = 1e-5
    Tem = torch.ones_like(target_entropy, requires_grad=True)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD([Tem], lr=lr)
    for epoch in range(n_iterations):
        input_weights = F.softmax(input_logits/Tem[:,None,:,None,None], dim=-1)
        input_weights = input_weights * attention_mask[None, :, None, :, None]
        entropy = -input_weights * (input_weights+eps).log()
        entropy = ((entropy.sum(dim=-1) * attention_mask[None,:,None,:]).sum(-1) / (attention_mask[None,:,None,:].sum(-1) + eps)).mean(dim=1)
        loss=criterion(entropy, target_entropy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print("Epoch", epoch)
        # print("loss", loss)
    
    return Tem




if __name__=="__main__":
    input_logits = torch.randn(14, 16, 32, 144, 100, device="cuda:1")
    target_entropy = torch.ones(14, 32, device="cuda:1")
    attention_mask = torch.ones(16, 144, device="cuda:1")
    Tem = get_temperature(input_logits=input_logits, target_entropy=target_entropy, attention_mask=attention_mask)
    print(Tem)


