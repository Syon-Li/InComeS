# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 13:57:21 2023

@author: Li_Sh
"""


# from editor_new import BaseEditor_new

from EasyEdit.easyeditor import BaseEditor, ZsreDataset, CounterFactDataset
from EasyEdit.easyeditor import (MEMITHyperParams, MENDHyperParams, SERACHparams, FTHyperParams, GraceHyperParams, LoRAHyperParams, ROMEHyperParams,
DINMHyperParams, EMMETHyperParams, IKEHyperParams, KNHyperParams, MALMENHyperParams, MELOHyperParams, PMETHyperParams, QLoRAHyperParams, 
R_ROMEHyperParams, WISEHyperParams)
from EasyEdit.easyeditor.evaluate import evaluate_utils, evaluate
from EasyEdit.easyeditor.models.ike.util import encode_ike_facts
from EasyEdit.easyeditor.util import nethook
from EasyEdit.easyeditor.models.grace import GRACE
from sentence_transformers import SentenceTransformer
from eval_utils import SeracRewriteExecutor_new, BaseEditor_new
from utils import _chunks
import json
import torch
import copy
import numpy as np
import argparse
import os
from huggingface_hub import login
login(token = "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")


os.environ['TOKENIZERS_PARALLELISM'] = "false"

# def compute_locality_quality(
#     model,
#     model_name,
#     hparams, 
#     tok,
#     locality_key: str,
#     prompt: typing.Union[str, typing.List[str]],
#     locality_ground_truth: typing.Union[str, typing.List[str]],
#     device,
# ) -> typing.Dict:

#     if 't5' in model_name.lower():
#         loc_tokens = evaluate_utils.test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, locality_ground_truth, device, locality=True)
#     else:
#         loc_tokens = evaluate_utils.test_prediction_acc(model, tok, hparams, prompt, locality_ground_truth, device, locality=False, vanilla_generation=hparams.alg_name=='GRACE')

#     if type(loc_tokens) is not list:
#         loc_tokens = [loc_tokens,]

#     ret = {
#         f"{locality_key}_output": loc_tokens
#     }
#     return ret

# # change function so locality was calculated based on ground truth
# evaluate.compute_locality_quality = compute_locality_quality




# def split_epsilons_in_half(self, nearest_key, smallest_distance):
#     # print(smallest_distance, self.epsilons[nearest_key])
#     # self.epsilons[nearest_key] = self.epsilons[nearest_key].to(torch.float)
#     self.epsilons[nearest_key] = (smallest_distance.item() / 2) - 1e-5 # Cut nearest epsilon in half
#     self.epsilons[-1] = smallest_distance.item() / 2 # Cut new epsilon in half  

# GRACE.GRACEAdapter.split_epsilons_in_half = split_epsilons_in_half






def main():
    # # editing_method = "SERAC"
    # editing_method = "MEMIT"
    # # editing_method = "FT-L"
    # # editing_method = "FT-M"
    # # editing_method = "MEND"
    # # editing_method = "GRACE"
    # # editing_method = "LoRA"
    # model_id = "EleutherAI/gpt-j-6B"
    # # model_id = "meta-llama/Llama-2-7b-hf"
    # model_name = model_id.rpartition("/")[-1]
    # ds_name = "zsre_mend_eval"
    # # ds_name = "counterfact-edit"
    # batch_edit = True
    # sequential_edit = False
    # bsz = 100
    # transformers_cache = "./transformers_cache"




    parser = argparse.ArgumentParser(description='Model editing baselines.')
    parser.add_argument('--method', type=str, help='the editing method to use')
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-3B", help='the model to use')
    parser.add_argument("--dataset", type=str, default='counterfact', help="the dataset to use")
    parser.add_argument("--batch_edit", type=int, choices=[0,1], default=0, help="whether to conduct batch edit")
    parser.add_argument("--sequential_edit", type=int, choices=[0,1], default=0, help="whether to conduct sequential")
    parser.add_argument('--bsz', type=int, default=30, help='the editing batch size to use') 
    parser.add_argument('--device', type=int, default=0, help='the device to use')  
    parser.add_argument('--wrt', type=int, default=0, help='whether to write the result file')  
    args = parser.parse_args()

    # torch.cuda.set_device(f"cuda:{args.device}")

    editing_method = args.method
    model_id = args.model
    batch_edit = True if args.batch_edit else False
    sequential_edit = True if args.sequential_edit else False
    model_name = model_id.rpartition("/")[-1]
    bsz = args.bsz
    wrt = args.wrt
    transformers_cache = "./transformers_cache"
    if args.dataset == "zsre":
        ds_name = "zsre_mend_eval"
    elif args.dataset == "counterfact":
        ds_name = "counterfact-edit"
    else:
        ds_name = args.dataset
    
    print(model_id)


    if model_name == "Llama-2-7b-hf":
        model_name = "llama-7b"
    
    if "llama-3.2" in model_name.lower():
        # nethook.get_parameter = get_parameter
        model_name = "llama3.2-3b"
    
    if model_name == "TinyLlama_v1.1":
        model_name = "llama-7b"


    
    if editing_method == "ROME":
        hparams = ROMEHyperParams.from_hparams('./EasyEdit/hparams/{}/{}.yaml'.format(editing_method, model_name))
        if "llama" in model_name.lower():
            # hparams.v_loss_layer = 27
            hparams.v_loss_layer = 15
            hparams.model_name = model_id

    elif editing_method == "GRACE":
        if model_name == "gpt-j-6B":
            model_name = "gpt-j-6b"
        hparams = GraceHyperParams.from_hparams('./EasyEdit/hparams/{}/{}.yaml'.format(editing_method, model_name))
        hparams.n_iter = 100
        # hparams.eps = 3
        if "llama" in model_name.lower():
            # print(hparams.inner_params)
            hparams.inner_params = ['model.layers[13].mlp.down_proj.weight']
            hparams.model_name = model_id       
    
    elif editing_method == "IKE":
        hparams = IKEHyperParams.from_hparams('./EasyEdit/hparams/{}/{}.yaml'.format(editing_method, model_name))
        hparams.sentence_model_name = hparams.sentence_model_name.rpartition("/")[-1]
        if "counterfact" in ds_name.lower():
            train_ds = CounterFactDataset('./Editing_data/counterfact/counterfact-train.json')
        elif "zsre" in ds_name.lower():
            train_ds = ZsreDataset('./Editing_data/zsre/zsre_mend_train_10000.json')
        else:
            train_ds = ZsreDataset('./Editing_data/zsre/zsre_mend_train_10000.json')
        # Initialize SentenceTransformer model
        sentence_model = SentenceTransformer(hparams.sentence_model_name)
        # Generate and save sentence embeddings
        encode_ike_facts(sentence_model, train_ds, hparams)
        if "llama" in model_name.lower():
            hparams.model_name = model_id
    
    # elif editing_method == "DINM":
    #     hparams = DINMHyperParams.from_hparams('./EasyEdit/hparams/{}/{}.yaml'.format(editing_method, model_name))

    elif editing_method == "R-ROME":
        hparams = R_ROMEHyperParams.from_hparams('./EasyEdit/hparams/{}/{}.yaml'.format(editing_method, model_name))
        if "llama" in model_name.lower():
            # hparams.v_loss_layer = 27
            hparams.v_loss_layer = 15
            hparams.model_name = model_id

    elif editing_method == "WISE":
        hparams = WISEHyperParams.from_hparams('./EasyEdit/hparams/{}/{}.yaml'.format(editing_method, model_name))
        # hparams.inner_params =

    elif editing_method == "KN":
        hparams = KNHyperParams.from_hparams('./EasyEdit/hparams/{}/{}.yaml'.format(editing_method, model_name))
        if "llama" in model_name.lower():
            hparams.model_name = model_id
        hparams.batch_size = 1
        hparams.steps = 1

    elif editing_method == "MEMIT":
        hparams = MEMITHyperParams.from_hparams('./EasyEdit/hparams/{}/{}.yaml'.format(editing_method, model_name))
        if "llama" in model_name.lower():
            # hparams.v_loss_layer = 27
            hparams.v_loss_layer = 15
            hparams.model_name = model_id

    elif editing_method == "EMMET":
        if "llama3.2" in model_name.lower():
            hparams = EMMETHyperParams.from_hparams('./EasyEdit/hparams/{}/{}.yaml'.format(editing_method, "llama-7b"))
            hparams.v_loss_layer = 15
            hparams.model_name = model_id
            hparams.lm_head_module = "model.embed_tokens"
        else:
            hparams = EMMETHyperParams.from_hparams('./EasyEdit/hparams/{}/{}.yaml'.format(editing_method, model_name))
        
    elif editing_method == "PMET":
        if "llama3.2" in model_name.lower():
            hparams = PMETHyperParams.from_hparams('./EasyEdit/hparams/{}/{}.yaml'.format(editing_method, "llama-7b"))
            hparams.v_loss_layer = 15
            hparams.model_name = model_id
            hparams.lm_head_module = "model.embed_tokens"
        else:
            hparams = PMETHyperParams.from_hparams('./EasyEdit/hparams/{}/{}.yaml'.format(editing_method, model_name))

    # elif editing_method == "MALMEN":
    #     hparams = MALMENHyperParams.from_hparams('./EasyEdit/hparams/{}/{}.yaml'.format(editing_method, model_name))

    elif editing_method == "QLoRA":
        hparams = QLoRAHyperParams.from_hparams('./EasyEdit/hparams/{}/{}.yaml'.format(editing_method, model_name))
        
    elif editing_method == "MEND":
        hparams = MENDHyperParams.from_hparams('./EasyEdit/hparams/{}/{}.yaml'.format(editing_method, model_name))
        if model_name == "gpt2-xl":
            # if ds_name == "zsre_mend_eval":
            #     ds_str = "zsre"
            # else:
            #     ds_str = "counterfact"
            # keep_original_weight = True
            hparams.archive = "./results/{}_{}_{}/models/{}/{}".format(editing_method, model_name, args.dataset, editing_method, model_name)
        else:
            hparams.archive = "./results/{}_{}/mend-10tok-gpt-j-6b.pt".format(editing_method, model_name)
        hparams.tokenizer_name = model_id   
        # hparams.device = 0
        
    elif editing_method == "SERAC":
        if "llama3.2" in model_name.lower():
            hparams = SERACHparams.from_hparams('./EasyEdit/hparams/{}/{}.yaml'.format(editing_method, "llama-7b"))
            hparams.model_name = model_id
            hparams.model_class = "AutoModelForCausalLM"
            hparams.tokenizer_class = "AutoTokenizer"
        else:
            hparams = SERACHparams.from_hparams('./EasyEdit/hparams/{}/{}.yaml'.format(editing_method, model_name))
            if ds_name == "zsre_mend_eval":
                ds_str = "zsre"
            elif ds_name == "counterfact-edit":
                ds_str = "counterfact"
            else:
                ds_str = "zsre"
            hparams.archive = "./results/{}_{}_{}/models/{}/{}".format(editing_method, model_name, ds_str, editing_method, model_name)
        hparams.tokenizer_name = hparams.tokenizer_name.rpartition("/")[-1]
        hparams.cls_name = hparams.cls_name.rpartition("/")[-1]
        if "gpt-j" in model_id:
            hparams.small_name = "anton-l/gpt-j-tiny-random"
        elif "Llama" in model_id:
            hparams.small_name = "meta-llama/Llama-3.2-1B-Instruct"
            
    elif editing_method == "FT-L":
        hparams = FTHyperParams.from_hparams('./EasyEdit/hparams/{}/{}.yaml'.format("FT", model_name))
        hparams.objective_optimization = "prompt_last"

    elif editing_method == "FT-M":
        hparams = FTHyperParams.from_hparams('./EasyEdit/hparams/{}/{}.yaml'.format("FT", model_name))
        hparams.objective_optimization = "target_new"
        if "llama" in model_name.lower():
            hparams.layers = [13]
            hparams.model_name = model_id

    elif editing_method == "LoRA":
        hparams = LoRAHyperParams.from_hparams('./EasyEdit/hparams/{}/{}.yaml'.format(editing_method, model_name))
        # hparams.lora_type = "lora"
        # hparams.lr = 1e-3
        if "llama" in model_name.lower():
            hparams.model_name = model_id

    # if model_name == "Llama-2-7b-hf":
        # hparams.model_parallel = True



    prompts, ground_truth, target_new, subjects = [], [], [], []
    rephrase_prompts = []
    locality_inputs, portability_inputs = {}, {}
    if ds_name == "counterfact-edit":
        with open("./Editing_data/counterfact/counterfact-edit.json") as f:
            ds = json.load(f)
        requests = ds[:]
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
        requests = ds[:10]
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
        for record in ds[:]:
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
                for record in ds[:]:
                    subject = record["requested_rewrite"]["subject"]
                    subjects.append(subject)
                    portability_inputs["one hop"]["prompt"].append(record["portability"]["New Question"])
                    portability_inputs["one hop"]["ground_truth"].append(record["portability"]["New Answer"])
                    prompts.append(record["requested_rewrite"]["prompt"].format(subject))
                    target_new.append(record["requested_rewrite"]["target_new"]["str"])
                    ground_truth.append(record["requested_rewrite"]["target_true"]["str"])        
            elif "zsre" in path:
                for record in ds[:]:
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
                for record in ds[:]:
                    subject = record["requested_rewrite"]["subject"]
                    al_subject = record["alternative_subject"]
                    subjects.append(subject)
                    portability_inputs["subject replacement"]["prompt"].append(record["requested_rewrite"]["prompt"].format(al_subject))
                    portability_inputs["subject replacement"]["ground_truth"].append(record["requested_rewrite"]["target_new"]["str"])
                    prompts.append(record["requested_rewrite"]["prompt"].format(subject))
                    target_new.append(record["requested_rewrite"]["target_new"]["str"])
                    ground_truth.append(record["requested_rewrite"]["target_true"]["str"])
            else:
                for record in ds[:]:
                    if record["subject"] in record["src"]:
                        subjects.append(record["subject"])
                        portability_inputs["subject replacement"]["prompt"].append(record["alter_subject_question"])
                        portability_inputs["subject replacement"]["ground_truth"].append(record["alt"])
                        prompts.append(record["src"])
                        target_new.append(record["alt"])
                        ground_truth.append(record["answers"][0])





    print(len(prompts), editing_method, ds_name, batch_edit, sequential_edit)

    if editing_method not in ["GRACE", "LoRA", "FT-M"]:
        hparams.fp16 = True
    hparams.device = args.device
    hparams.model_name = model_id
    # hparams.model_parallel = True
    if batch_edit:
        hparams.batch_size = bsz
    else:
        if not sequential_edit:
            bsz = len(prompts) # actual chunk size 

    editor = BaseEditor(hparams)
    # editor = BaseEditor_new(hparams)

    if editor.alg_name.lower() == "serac":
        editor.apply_algo = SeracRewriteExecutor_new().apply_to_model


    if batch_edit:
        metrics, edited_model, weights_copy = editor.batch_edit(
            prompts=prompts,
            target_new=target_new,
            ground_truth=ground_truth,
            rephrase_prompts=rephrase_prompts if len(rephrase_prompts)>0 else None,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            subject=subjects,
            # keep_original_weight=True,
            verbose=True,
        )
        all_metrics = metrics
        # print(weights_copy)
    else:
        if sequential_edit and editing_method == "GRACE":
            device = torch.device(f'cuda:{hparams.device}')
            origin_grace = GRACE.GRACE(model=editor.model, config=hparams, device=device)
            original_layer = copy.deepcopy(origin_grace.original_layer)
            # print(original_layer)

        rephrase_prompts_iter = _chunks(rephrase_prompts, bsz)
        locality_inputs_iter = _chunks(locality_inputs, bsz)
        portability_inputs_iter = _chunks(portability_inputs, bsz)
        all_metrics = []
        for prompts_chunk, target_new_chunk, ground_truth_chunk, subjects_chunk in \
        zip(_chunks(prompts, bsz), _chunks(target_new, bsz), _chunks(ground_truth, bsz), _chunks(subjects, bsz)):
            rephrase_prompts_chunk = next(rephrase_prompts_iter) if len(rephrase_prompts)>0 else None
            locality_inputs_chunk = next(locality_inputs_iter) if len(locality_inputs)>0 else None
            portability_inputs_chunk = next(portability_inputs_iter) if len(portability_inputs)>0 else None
            # print(len(prompts_chunk))
            # print(portability_inputs_chunk)
            metrics, edited_model, weights_copy = editor.edit(
                prompts=prompts_chunk,
                target_new=target_new_chunk,
                ground_truth=ground_truth_chunk,
                # prompts_eval=prompts_chunk,
                # target_new_eval=target_new_chunk,
                # ground_truth_eval=ground_truth_chunk,   
                rephrase_prompts=rephrase_prompts_chunk,
                locality_inputs=locality_inputs_chunk,
                portability_inputs=portability_inputs_chunk,
                subject=subjects_chunk,
                sequential_edit=sequential_edit,
                train_ds=train_ds if editing_method == "IKE" else None,
                verbose=False,
            )

            if sequential_edit:
                # only sequential edit for a given chunk needs to do this, non sequential edit roll-back happens inside the editor.edit()
                if editor.alg_name == 'KN' or editor.alg_name == 'GRACE' or editor.alg_name == 'WISE':
                    if editor.alg_name == 'GRACE':
                        # deliver the original_layer for resetting
                        edited_model.original_layer = original_layer
                        # print(edited_model.layer)
                    with torch.no_grad():
                        weights_copy()
                elif editor.alg_name == 'LoRA' or editor.alg_name == 'QLoRA':
                    edited_model.unload()
                    del editor.model.peft_config
                elif editor.alg_name == 'MELO':
                    editor.model = edited_model
                elif editor.alg_name == 'LoRA' or editor.alg_name == 'QLoRA':
                    editor.model = edited_model
                else:
                    with torch.no_grad():
                        for k, v in weights_copy.items():
                            nethook.get_parameter(editor.model, k)[...] = v.to(f"cuda:{editor.hparams.device}")
            # print(metrics)
            all_metrics.extend(metrics)




    rewrite_acc, rephrase_acc = 0, 0
    locality, portability = {}, {}
    for i,metric in enumerate(all_metrics):
        if isinstance(metric["post"]["rewrite_acc"], list):
            rewrite_acc += metric["post"]["rewrite_acc"][0]
        else:
            rewrite_acc += metric["post"]["rewrite_acc"]

        if "rephrase_acc" in metric["post"].keys():
            if isinstance(metric["post"]["rephrase_acc"], list):
                rephrase_acc += metric["post"]["rephrase_acc"][0]
            else:
                rephrase_acc += metric["post"]["rephrase_acc"]

        if "locality" in metric["post"].keys():
            for key,value in metric["post"]["locality"].items():
                if key not in locality.keys():
                    locality[key] = 0
                if "_output" in key:
                    locality[key] += np.mean(np.equal(metric["pre"]["locality"][key], value)).item()
                elif "_acc" in key:
                    if isinstance(value, list):
                        locality[key] += value[0]
                    else:
                        locality[key] += value
        
        if "portability" in metric["post"].keys():
            for key,value in metric["post"]["portability"].items():
                if key not in portability.keys():
                    portability[key] = 0
                if isinstance(value, list):
                    portability[key] += value[0]
                else:
                    portability[key] += value
                    
    results = {"rewrite_acc": rewrite_acc/len(all_metrics), "rephrase_acc": rephrase_acc/len(all_metrics)}
    for k,v in locality.items():
        locality[k] = v/len(all_metrics)
    for k,v in portability.items():
        portability[k] = v/len(all_metrics)
    results["locality"] = locality
    results["portability"] = portability
    

    run_config = {
                "model_id": model_id,
                "editing model": editing_method,
                "batch_size": bsz,
                "batch_edit": batch_edit,
                "sequential_edit": sequential_edit,
                "dataset": ds_name,
                "requests_size": len(prompts),
                }
    print(run_config)
    print(results)

    if wrt:
        model_name = model_id.rpartition("/")[-1]
        os.makedirs(f"./Experiment_results/{ds_name}", exist_ok=True)
        with open(f"./Experiment_results/{ds_name}/{model_name}-{editing_method}-{bsz}.json","w") as f:
            json.dump(run_config, f)
            f.write("\n")
            json.dump(results, f)





if __name__=="__main__":
    main()