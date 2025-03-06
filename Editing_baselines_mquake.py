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
from EasyEdit.easyeditor.editors.batch_editor import BatchEditor
from EasyEdit.easyeditor.evaluate import evaluate_utils, evaluate
from EasyEdit.easyeditor.models.ike.util import encode_ike_facts
from EasyEdit.easyeditor.util import nethook
from EasyEdit.easyeditor.models.grace import GRACE
from EasyEdit.easyeditor.editors.utils import _prepare_requests
from EasyEdit.easyeditor.evaluate.evaluate import compute_edit_quality
from sentence_transformers import SentenceTransformer
from eval_utils import BaseEditor_new, compute_edit_quality_hop, SeracRewriteExecutor_new
from utils import _chunks
import json
from copy import deepcopy
import torch
import copy
import numpy as np
import typing
import argparse
import os


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
    parser.add_argument("--multiple_edit", type=int, choices=[0,1], default=0, help="whether to edit multiple edits")
    parser.add_argument('--bsz', type=int, default=30, help='the multiple edits size to use') 
    parser.add_argument('--edits', type=int, default=2, help='the hop number of the edit') 
    parser.add_argument('--device', type=int, default=0, help='the device to use')  
    parser.add_argument('--wrt', type=int, default=0, help='whether to write the result file')  
    args = parser.parse_args()

    # torch.cuda.set_device(f"cuda:{args.device}")

    editing_method = args.method
    model_id = args.model
    multiple_edit = args.multiple_edit
    model_name = model_id.rpartition("/")[-1]
    bsz = args.bsz
    edits = args.edits
    wrt = args.wrt
    transformers_cache = "./transformers_cache"
    ds_name = "mquake"
    
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
        # hparams.eps = 30
        if "llama" in model_name.lower():
            # print(hparams.inner_params)
            hparams.inner_params = ['model.layers[13].mlp.down_proj.weight']
            hparams.model_name = model_id        
    
    elif editing_method == "IKE":
        hparams = IKEHyperParams.from_hparams('./EasyEdit/hparams/{}/{}.yaml'.format(editing_method, model_name))
        hparams.sentence_model_name = hparams.sentence_model_name.rpartition("/")[-1]
        train_ds = CounterFactDataset('./Editing_data/counterfact/counterfact-train.json')
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
        # if "llama3.2" in model_name.lower():
        #     hparams.v_loss_layer = 15
        #     hparams.model_name = model_id
        #     hparams.lm_head_module = "model.embed_tokens"   

    elif editing_method == "PMET":
        if "llama3.2" in model_name.lower():
            hparams = PMETHyperParams.from_hparams('./EasyEdit/hparams/{}/{}.yaml'.format(editing_method, "llama-7b"))
            hparams.v_loss_layer = 15
            hparams.model_name = model_id
            hparams.lm_head_module = "model.embed_tokens"
        else:
            hparams = PMETHyperParams.from_hparams('./EasyEdit/hparams/{}/{}.yaml'.format(editing_method, model_name))
        # if "llama3.2" in model_name.lower():
        #     hparams.v_loss_layer = 15
        #     hparams.model_name = model_id
        #     hparams.lm_head_module = "model.embed_tokens"            

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
        hparams.device = 0
        
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
        print(hparams)

    elif editing_method == "LoRA":
        hparams = LoRAHyperParams.from_hparams('./EasyEdit/hparams/{}/{}.yaml'.format(editing_method, model_name))
        # hparams.lora_type = "lora"
        # hparams.lr = 1e-3
        if "llama" in model_name.lower():
            hparams.model_name = model_id

    # if model_name == "Llama-2-7b-hf":
        # hparams.model_parallel = True



    prompts, ground_truth, target_new, subjects = [], [], [], []
    if ds_name.lower() == "mquake":
        with open("./Editing_data/MQuAKE/MQuAKE-CF-3k-v2.json") as f:
            ds = json.load(f)

        # print(ds, len(ds))
        requests = ds[:]

        prompts, subjects, target_new, ground_truth = [], [], [], []
        prompts_hop, target_new_hop, ground_truth_hop = [], [], []
        for record in requests:
            if len(record["requested_rewrite"]) == edits:
                for item in [prompts, subjects, target_new, ground_truth]:
                    item.append([])
                
                for rew in record["requested_rewrite"]:
                    subjects[-1].append(rew["subject"])
                    target_new[-1].append(rew["target_new"]["str"])
                    ground_truth[-1].append(rew["target_true"]["str"])
                    prompts[-1].append(rew["prompt"].format(rew["subject"]))
                
                prompts_hop.append(record["questions"])
                # target_new_hop.append(record["new_answer"])
                # ground_truth_hop.append(record["answer"])

                new_answer = [record["new_answer"]]
                new_answer.extend(record["new_answer_alias"])
                target_new_hop.append(new_answer)

                answer = [record["answer"]]
                answer.extend(record["answer_alias"])
                ground_truth_hop.append(answer)    



    # print(prompts)
    # print(prompts_hop)
    


    if editing_method not in ["GRACE", "LoRA", "FT-M"]:
        hparams.fp16 = True
    hparams.device = args.device
    hparams.model_name = model_id
    if not multiple_edit:
        bsz = 1

    editor = BaseEditor_new(hparams)
    if editor.alg_name.lower() == "serac" and "llama" in model_name.lower():
        editor.apply_algo = SeracRewriteExecutor_new().apply_to_model

    print(len(prompts), editing_method, ds_name, multiple_edit)
    # print(editor.model)

    
    if BatchEditor.is_batchable_method(editor.alg_name):
        all_metrics = []
        for prompts_chunk, target_new_chunk, ground_truth_chunk, subjects_chunk, \
            prompts_hop_chunk, target_new_hop_chunk, ground_truth_hop_chunk in \
            zip(_chunks(prompts, bsz), _chunks(target_new, bsz), _chunks(ground_truth, bsz), _chunks(subjects, bsz), 
                _chunks(prompts_hop, bsz), _chunks(target_new_hop, bsz), _chunks(ground_truth_hop, bsz)):
            
            flattened_prompts = [item for sublist in prompts_chunk for item in sublist]
            flattened_target_new = [item for sublist in target_new_chunk for item in sublist]
            flattened_ground_truth = [item for sublist in ground_truth_chunk for item in sublist]
            flattened_subjects = [item for sublist in subjects_chunk for item in sublist]

            # deduplication
            if multiple_edit:
                contexts = {}
                for i, (fla_prompt, fla_tn) in enumerate(zip(flattened_prompts, flattened_target_new)):
                    contexts.update({fla_prompt + " " + fla_tn:i})
                unique_contexts = list(contexts.keys())
                uni_flattened_prompts, uni_flattened_target_new, uni_flattened_ground_truth, uni_flattened_subjects = [], [], [], []
                for kn in unique_contexts:
                    uni_flattened_prompts.append(flattened_prompts[contexts[kn]])
                    uni_flattened_target_new.append(flattened_target_new[contexts[kn]])
                    uni_flattened_ground_truth.append(flattened_ground_truth[contexts[kn]])
                    uni_flattened_subjects.append(flattened_subjects[contexts[kn]])
            else:
                uni_flattened_prompts = flattened_prompts
                uni_flattened_target_new = flattened_target_new
                uni_flattened_ground_truth = flattened_ground_truth
                uni_flattened_subjects = flattened_subjects               
            
            # editor.hparams.batch_size = len(uni_flattened_prompts)
            # print(editor.hparams)
            metrics, edited_model, weights_copy = editor.batch_edit(
                prompts=uni_flattened_prompts,
                target_new=uni_flattened_target_new,
                ground_truth=uni_flattened_ground_truth,
                subject=uni_flattened_subjects,
                prompts_eval=prompts_hop_chunk,
                target_new_eval=target_new_hop_chunk,
                ground_truth_eval=ground_truth_hop_chunk,
                keep_original_weight=True,
                verbose=True,
            )
            all_metrics.extend(metrics)
            # print(weights_copy)
            
    else:
        if editing_method == "GRACE":
            device = torch.device(f'cuda:{hparams.device}')
            origin_grace = GRACE.GRACE(model=editor.model, config=hparams, device=device)
            original_layer = copy.deepcopy(origin_grace.original_layer)

        all_metrics = []
        for prompts_chunk, target_new_chunk, ground_truth_chunk, subjects_chunk, \
            prompts_hop_chunk, target_new_hop_chunk, ground_truth_hop_chunk in \
            zip(_chunks(prompts, bsz), _chunks(target_new, bsz), _chunks(ground_truth, bsz), _chunks(subjects, bsz), 
                _chunks(prompts_hop, bsz), _chunks(target_new_hop, bsz), _chunks(ground_truth_hop, bsz)):
            
            flattened_prompts = [item for sublist in prompts_chunk for item in sublist]
            flattened_target_new = [item for sublist in target_new_chunk for item in sublist]
            flattened_ground_truth = [item for sublist in ground_truth_chunk for item in sublist]
            flattened_subjects = [item for sublist in subjects_chunk for item in sublist]

            # deduplication
            if multiple_edit:
                contexts = {}
                for i, (fla_prompt, fla_tn) in enumerate(zip(flattened_prompts, flattened_target_new)):
                    contexts.update({fla_prompt + " " + fla_tn:i})
                unique_contexts = list(contexts.keys())
                uni_flattened_prompts, uni_flattened_target_new, uni_flattened_ground_truth, uni_flattened_subjects = [], [], [], []
                for kn in unique_contexts:
                    uni_flattened_prompts.append(flattened_prompts[contexts[kn]])
                    uni_flattened_target_new.append(flattened_target_new[contexts[kn]])
                    uni_flattened_ground_truth.append(flattened_ground_truth[contexts[kn]])
                    uni_flattened_subjects.append(flattened_subjects[contexts[kn]])
            else:
                uni_flattened_prompts = flattened_prompts
                uni_flattened_target_new = flattened_target_new
                uni_flattened_ground_truth = flattened_ground_truth
                uni_flattened_subjects = flattened_subjects 

            metrics, edited_model, weights_copy = editor.edit(
                prompts=uni_flattened_prompts,
                target_new=uni_flattened_target_new,
                ground_truth=uni_flattened_ground_truth,
                subject=uni_flattened_subjects,
                sequential_edit=True,
                train_ds=train_ds if editing_method == "IKE" else None,
                verbose=False,
            )
            # print(metrics)

            requests_hop = _prepare_requests(prompts_hop_chunk, target_new_hop_chunk, ground_truth_hop_chunk)
            for i, request in enumerate(requests_hop):

                metrics = {
                    'case_id': i,
                    "requested_rewrite": request,
                    "post": compute_edit_quality_hop(edited_model, editor.model_name, editor.hparams, editor.tok, request, editor.hparams.device),
                }

                all_metrics.append(metrics)

            if editor.alg_name == 'KN' or editor.alg_name == 'GRACE' or editor.alg_name == 'WISE':
                if editor.alg_name == 'GRACE':
                    # deliver the original_layer for resetting
                    edited_model.original_layer = original_layer
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

            




    rewrite_acc = 0
    for i,metric in enumerate(all_metrics):
        if isinstance(metric["post"]["rewrite_acc"], list):
            rewrite_acc += metric["post"]["rewrite_acc"][0]
        else:
            rewrite_acc += metric["post"]["rewrite_acc"]

                    
    results = {"rewrite_acc": rewrite_acc/len(all_metrics)}
    

    run_config = {
                "model_id": model_id,
                "editing model": editing_method,
                "batch_size": bsz,
                "edits":edits,
                "multiple_edit": multiple_edit,
                "dataset": ds_name,
                "requests_size": len(prompts),
                }
    print(all_metrics, len(all_metrics))
    print(run_config)
    print(results)

    if wrt:
        model_name = model_id.rpartition("/")[-1]
        os.makedirs(f"./Experiment_results/{ds_name}", exist_ok=True)
        with open(f"./Experiment_results/{ds_name}/{model_name}-{editing_method}-{edits}-{bsz}.json","w") as f:
            json.dump(run_config, f)
            f.write("\n")
            json.dump(results, f)





if __name__=="__main__":
    main()