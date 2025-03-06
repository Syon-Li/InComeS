from typing import List, Dict, Optional, Union
import typing
import torch
import json
import random
import transformers
from tqdm import tqdm
import numpy as np
from time import time
from copy import deepcopy
from EasyEdit.easyeditor import BaseEditor, LOG
from EasyEdit.easyeditor.util.hparams import HyperParams
from EasyEdit.easyeditor.editors.batch_editor import BatchEditor
from EasyEdit.easyeditor.util import nethook
from EasyEdit.easyeditor.editors.utils import _prepare_requests, summary_metrics
from utils import _chunks
from EasyEdit.easyeditor.models.melo.melo import LORA
from EasyEdit.easyeditor.models.serac import SeracRewriteExecutor
from EasyEdit.easyeditor.models.serac.serac_main import SERACHparams, SERAC
from transformers import AutoTokenizer
from EasyEdit.easyeditor.evaluate import compute_edit_quality, compute_rewrite_or_rephrase_quality, compute_icl_edit_quality, compute_sent_metric



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
            if subject in prompt:
                requests.append({"prompt":prompt, "target_new":target_new, "ground_truth":ground_truth, "rephrase_prompt":rephrase_prompt, 
                                "locality":locality, "subject":subject, "knowledge":prompt.strip() + ' ' + target_new.strip() + "."})
    
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
            requests.append({"prompt":inputs.strip(), "target_new":edit.strip(), "knowledge": f"{edit.strip()}"})
        else:
            if type(inputs[0])==list:
                for txt in inputs:
                    lines = txt[0].splitlines()
                    for line in lines:
                        if len(line)>1:
                            if line[0] == txt[1]:
                                target_new = ""
                                for i,word in enumerate(line.split()):
                                    if i!=0:
                                        target_new += " " + word
                                requests.append({"prompt":lines[0].strip(), "target_new":target_new.strip(), "knowledge": edit})
                                break
            else:
                lines = inputs[0].splitlines()
                for line in lines:
                    if line[0] == inputs[1]:
                        target_new = ""
                        for i,word in enumerate(line.split()):
                            if i!=0:
                                target_new += " " + word
                        requests.append({"prompt":lines[0].strip(), "target_new":target_new.strip(), "knowledge": edit})
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
                                 "subject":subject, "knowledge":old_prompt.strip() + ' ' + target_new.strip() + "."})
        else:
            for record in ds[:]:
                subject = record["subject"]
                prompt = record["alter_subject_question"]
                old_prompt = record["src"]
                target_new = record["alt"]
                ground_truth = record["answers"][0]
                requests.append({"prompt":prompt, "target_new":target_new, "ground_truth":ground_truth, 
                                 "subject":subject, "knowledge":old_prompt.strip() + ' ' + target_new.strip() + "."})

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
                             "subject":subject, "knowledge":old_prompt.strip() + ' ' + target_new.strip() + "."})
    
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
                                 "subject":subject, "knowledge":old_prompt.strip() + ' ' + old_target_new.strip() + "."})
        else:
            for record in ds[:]:
                subject = record["subject"]
                prompt = record["portability"]["New Question"]
                target_new = record["portability"]["New Answer"]
                old_prompt = record["src"]
                old_target_new = record["alt"]
                ground_truth = record["pred"]
                requests.append({"prompt":prompt, "target_new":target_new, "ground_truth":ground_truth, 
                                 "subject":subject, "knowledge":old_prompt.strip() + ' ' + old_target_new.strip() + "."})

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








def test_prediction_acc_gist(model, tok, knowledge:str, prompts:str, 
                             targets:str, gist_pool: Dict, method:str, 
                             locality=False, gist_pool_idx=None, loss_record={}):
    if isinstance(prompts, str):
        knowledge,prompts,targets = [knowledge,], [prompts,], [targets,]

    if method=="ICE":
        k_prompt_target = [k + " " + prompt + " " + target for k, prompt, target in zip(knowledge,prompts,targets)]
        k_prompt = [k + " " + prompt for k, prompt in zip(knowledge,prompts)]   
    else:
        k_prompt_target = [prompt + " " + target for prompt, target in zip(prompts,targets)]
        k_prompt = prompts 
    # elif input_mode=="gist":
    #     # k_prompt_target = [k + f' {tok.additional_special_tokens[-1]}{tok.additional_special_tokens[-1]} ' + prompt + " " + target for k, prompt, target in zip(knowledge,prompts,targets)]
    #     # k_prompt = [k + f' {tok.additional_special_tokens[-1]}{tok.additional_special_tokens[-1]} ' + prompt for k, prompt in zip(knowledge,prompts)]
    #     k_prompt_target = [k + f' {tok.additional_special_tokens[-1]} ' + prompt + " " + target for k, prompt, target in zip(knowledge,prompts,targets)]
    #     k_prompt = [k + f' {tok.additional_special_tokens[-1]} ' + prompt for k, prompt in zip(knowledge,prompts)]

    # print("k_prompt", k_prompt[0][-200:-1])
    # print("k_prompt_target", k_prompt_target[0][-200:-1])

    # k_prompt[0] = k_prompt[0][-200:-1]
    # k_prompt_target[0] = k_prompt_target[0][-203:-1]

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
    assert len(k_prompt_target_tok[0]) <= model.config.max_position_embeddings if "llama" in model.config.model_type.lower() else model.config.n_positions, "inputs exceed maximum allowed length"

    # if gist_pool is not None and gist_pool[0]["keys"].shape[0] > 1:
    if method=="gist":
        k_prompt_target_tok['input_ids'] = k_prompt_target_tok['input_ids'][:,1:]
        k_prompt_tok['input_ids'] = k_prompt_tok['input_ids'][:,1:]
        k_prompt_target_tok['attention_mask'] = k_prompt_target_tok['attention_mask'][:,1:]
        k_prompt_tok['attention_mask'] = k_prompt_tok['attention_mask'][:,1:]

    num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in k_prompt_tok['input_ids']]
    num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in k_prompt_target_tok['input_ids'].cpu()]
    prompt_len = [x+y for x,y in zip(num_pad_toks,num_prompt_toks)]


    model.eval()
    with torch.no_grad():
        if method=="gist":
            # model.gist_pool = gist_pool
            # model.gist_token_ids = tok.additional_special_tokens_ids[-1]
            outputs= model(**k_prompt_target_tok, 
                        gist_token_ids=tok.additional_special_tokens_ids[-1],
                        gist_pool=gist_pool,
                        gist_pool_idx=gist_pool_idx.to(model.device) if gist_pool_idx is not None else None,
                        extra_loss=loss_record,
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
        



def compute_edit_quality_new(record, model, tok, gist_pool, knowledge_set: List, method:str, gist_pool_idx=None, loss_record={}):
    
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
                                            gist_pool_idx=gist_pool_idx,
        )

    rephrase_acc = None
    if "rephrase_prompt" in record.keys():
        rephrase_acc = test_prediction_acc_gist(model=model, 
                                                tok=tok, 
                                                knowledge=knowledge, 
                                                prompts=record["rephrase_prompt"], 
                                                targets=record["target_new"],
                                                gist_pool=gist_pool, 
                                                loss_record=loss_record,
                                                method=method,
                                                )

    locality_output = {}
    if "locality" in record.keys():
        for key, value in record["locality"].items():
            loc_out = test_prediction_acc_gist(model=model, 
                                                tok=tok, 
                                                knowledge=knowledge, 
                                                prompts=value["prompt"], 
                                                targets=value["ground_truth"],
                                                gist_pool=gist_pool, 
                                                loss_record=loss_record,
                                                locality=True,
                                                method=method,
                                                )
            locality_output[key] = loc_out
    
    res.update({"rewrite_acc": rewrite_acc, "rephrase_acc": rephrase_acc if rephrase_acc else [0], "locality_output": locality_output})
    # print("res", res)
    
    return res




# def compute_edit_quality_single(records, model, tok, gist_pool, method:str):
#     if method == "ICE":
#         input_mode = "ICE"
#         model_mode = "original"
#     elif method == "gist":
#         input_mode = "gist"
#         model_mode = "gist"
#     elif method == "base":
#         input_mode = "original"
#         model_mode = "original" 

#     res_all = []
#     for i,record in enumerate(records):
#         knowledge = record["knowledge"]
#         res = compute_edit_quality_new(record=record, model=model, tok=tok, 
#                                        gist_pool=gist_pool, knowledge_set=[knowledge], input_mode=input_mode, 
#                                        model_mode=model_mode, keep_gist=False,)

#         # if i<5 and gist_pool is not None:
#         #     print(gist_pool[0]["keys"].shape)
        
#         if i<5 and input_mode=="ICE":
#             print(knowledge)

#         res_all.append(res)
    
#     return res_all




# def compute_edit_quality_batch(records, model, tok, gist_pool, knowledge_set:List, method:str,):

#     if method == "ICE":
#         input_mode = "ICE"
#         model_mode = "original"
#     elif method == "gist":
#         input_mode = "original"
#         model_mode = "gist"
#     elif method == "base":
#         input_mode = "original"
#         model_mode = "original"

#     res_all = []
#     if model_mode == "gist":
#         contexts = []
#         for knowledge in knowledge_set:
#             # contexts.append(knowledge + f" {tok.additional_special_tokens[-1]}{tok.additional_special_tokens[-1]} ")
#             contexts.append(knowledge + f" {tok.additional_special_tokens[-1]} ")
#         inputs = tok(contexts, padding=True, return_tensors="pt",).to(model.device)
#         model.eval()
    
#         with torch.no_grad():
#             # model.gist_pool = gist_pool
#             # model.gist_token_ids = tok.additional_special_tokens_ids[-1]
#             _ = model(**inputs, 
#                     gist_token_ids=tok.additional_special_tokens_ids[-1],
#                     gist_pool=gist_pool,
#                     use_cache=False,)
#             print(gist_pool[0]["keys"].shape)
    
#     if method == "ICE":
#         print("number of context", len(knowledge_set))
#     for i,record in enumerate(records):
#         # gist_pool_idx = torch.zeros(1, len(knowledge_set)+1) # plus one to consider the zero gist key and value 
#         # gist_pool_idx[0,i+1] = 1
#         gist_pool_idx = None
#         res = compute_edit_quality_new(record=record, model=model, tok=tok, 
#                                        gist_pool=gist_pool if gist_pool else None, knowledge_set=knowledge_set, input_mode=input_mode, 
#                                        gist_pool_idx=gist_pool_idx if gist_pool_idx is not None else None, model_mode=model_mode, keep_gist=True)
#         res_all.append(res)
    
#     return res_all




def eval_edit_quality(records, model, tok, method:str, knowledge_set:List = [], batch=False):

    gist_pool = {}
    for i in range(model.config.num_hidden_layers):
        gist_pool.update({i:{"keys":torch.tensor([]), "values":torch.tensor([])}})

    res_all = []
    if batch and method == "gist":
        contexts = []
        for knowledge in knowledge_set:
            contexts.append(knowledge + f" {tok.additional_special_tokens[-1]}")
        inputs = tok(contexts, padding=True, return_tensors="pt",).to(model.device)

        model.eval()
        with torch.no_grad():
            # model.gist_pool = gist_pool
            # model.gist_token_ids = tok.additional_special_tokens_ids[-1]
            _ = model(**inputs, 
                    gist_token_ids=tok.additional_special_tokens_ids[-1],
                    gist_pool=gist_pool,
                    use_cache=False,)
            print(gist_pool[0]["keys"].shape)
    
    if batch and method == "ICE":
        print("number of context", len(knowledge_set))
    for i,record in enumerate(records):
        # gist_pool_idx = torch.zeros(1, len(knowledge_set)+1) # plus one to consider the zero gist key and value 
        # gist_pool_idx[0,i+1] = 1
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
                    # model.gist_pool = gist_pool
                    # model.gist_token_ids = tok.additional_special_tokens_ids[-1]
                    _ = model(**inputs, 
                            gist_token_ids=tok.additional_special_tokens_ids[-1],
                            gist_pool=gist_pool,
                            use_cache=False,)
                    if i<5:
                        # print(contexts)
                        print(gist_pool[0]["keys"].shape)
            
            if method == "ICE" and i<5:
                print("number of context", len(knowledge_set))
            
        loss_record = {"Entropy":[]}
        res = compute_edit_quality_new(record=record, model=model, tok=tok, 
                                       gist_pool=gist_pool if gist_pool else None, knowledge_set=knowledge_set, method=method, 
                                       gist_pool_idx=gist_pool_idx if gist_pool_idx is not None else None, loss_record=loss_record)
        res_all.append(res)

        # print("Entropy", sum(loss_record["Entropy"])/len(loss_record["Entropy"]))
        # print("golden_idx:", i+1)
        print("Entropy:", loss_record["Entropy"])
    
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
        



def compute_edit_quality_hop(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device,
    eval_metric: str = 'token_em',
    test_generation = False
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    if isinstance(model,LORA):
        model=model.model
    # First, unpack rewrite evaluation record.
    target_new, ground_truth = (
        record[x] for x in ["target_new", "ground_truth"]
    )

    rewrite_prompts = record["prompt"]
    rephrase_prompts = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    r_acc = []
    for rewrite_prompt in rewrite_prompts:
        for tar_new in target_new:
            ret = compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                                    rewrite_prompt, tar_new, device=device, eval_metric=eval_metric)
            r_acc.append(ret["rewrite_acc"])
    ret["rewrite_acc"] = max(r_acc)

    ret['locality'] = {}
    ret['portability'] = {}

    return ret


        


class BaseEditor_new(BaseEditor):
    def __init__(self, hparams: HyperParams):
        super().__init__(hparams)
    
    def batch_edit_model(self,      
            prompts: List[str],
            target_new: List[str],
            ground_truth: Optional[List[str]] = None,
            target_neg: Optional[List[str]] = None,
            rephrase_prompts: Optional[List[str]] = None,
            locality_inputs: Optional[Dict] = None,
            portability_inputs: Optional[Dict] = None,
            sequential_edit=False,
            verbose=True,
            **kwargs
            ):
        
        assert len(prompts) == len(target_new)
        test_generation = kwargs['test_generation'] if 'test_generation' in kwargs.keys() else False
        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth,]
            else:
                assert len(ground_truth) == len(prompts)
        else: # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]


        assert BatchEditor.is_batchable_method(self.alg_name), f'The Method {self.alg_name} can not batch edit examples.'

        requests = _prepare_requests(prompts, target_new, ground_truth, target_neg, rephrase_prompts, locality_inputs, portability_inputs, **kwargs)
        

        assert hasattr(self.hparams, 'batch_size'), f'Method {self.alg_name} found, pls specify the batch_size....'
        all_metrics = []
        # for record_chunks in _chunks(requests, self.hparams.batch_size):
        start = time()

        edited_model, weights_copy = self.apply_algo(
            self.model,
            self.tok,
            requests,
            self.hparams,
            copy=False,
            return_orig_weights=True
        )
        exec_time = time() - start
        LOG.info(f"Execution editing took {exec_time}")

        start = time()
        chunk_metrics = []
        # for i, request in enumerate(requests_hop):

        #     metrics = {
        #         'case_id': i,
        #         "requested_rewrite": request,
        #         "time": exec_time,
        #         "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, test_generation=test_generation),
        #     }

        #     chunk_metrics.append(metrics)

        # if self.alg_name == 'KN' or self.alg_name == 'GRACE' or self.alg_name == 'WISE':
        #     with torch.no_grad():
        #         weights_copy()
        # elif self.alg_name == 'LoRA' or self.alg_name == 'QLoRA' or self.alg_name == 'DPO':
        #     edited_model.unload()
        #     del self.model.peft_config
        # elif self.alg_name == 'MELO':
        #     self.model = edited_model
        # else:
        #     with torch.no_grad():
        #         for k, v in weights_copy.items():
        #             nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

        # for i, request in enumerate(requests_hop):
        #     chunk_metrics[i]["pre"] = compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device, test_generation=test_generation)

        #     if verbose:
        #         LOG.info(
        #             f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {chunk_metrics[i]}"
        #         )

        # LOG.info(f"Evaluation took {time() - start}")
        all_metrics.extend(chunk_metrics)
        return all_metrics, edited_model, weights_copy   
    


    def batch_edit(self,
                   prompts: List[str],
                   target_new: List[str],
                   prompts_eval: List[str],
                   target_new_eval: List[str],
                   ground_truth_eval: Optional[List[str]] = None,    
                   ground_truth: Optional[List[str]] = None,
                   target_neg: Optional[List[str]] = None,
                   rephrase_prompts: Optional[List[str]] = None,               
                   locality_inputs: Optional[Dict] = None,
                   portability_inputs: Optional[Dict] = None,
                   sequential_edit=False,
                   verbose=True,
                   **kwargs
                   ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        """
        assert len(prompts) == len(target_new)
        test_generation = kwargs['test_generation'] if 'test_generation' in kwargs.keys() else False
        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth,]
            else:
                assert len(ground_truth) == len(prompts)
        else: # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]


        assert BatchEditor.is_batchable_method(self.alg_name), f'The Method {self.alg_name} can not batch edit examples.'

        requests = _prepare_requests(prompts, target_new, ground_truth, target_neg, rephrase_prompts, locality_inputs, portability_inputs, **kwargs)
        requests_hop = _prepare_requests(prompts_eval, target_new_eval, ground_truth_eval)

        self.hparams.batch_size = len(requests)
        assert hasattr(self.hparams, 'batch_size'), f'Method {self.alg_name} found, pls specify the batch_size....'
        all_metrics = []
        for record_chunks in _chunks(requests, self.hparams.batch_size):
            start = time()

            edited_model, weights_copy = self.apply_algo(
                self.model,
                self.tok,
                record_chunks,
                self.hparams,
                copy=False,
                return_orig_weights=True
            )
            exec_time = time() - start
            LOG.info(f"Execution editing took {exec_time}")

            start = time()
            chunk_metrics = []
            for i, request in enumerate(requests_hop):

                metrics = {
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_edit_quality_hop(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, test_generation=test_generation),
                }

                chunk_metrics.append(metrics)

            if self.alg_name == 'KN' or self.alg_name == 'GRACE' or self.alg_name == 'WISE':
                with torch.no_grad():
                    weights_copy()
            elif self.alg_name == 'LoRA' or self.alg_name == 'QLoRA' or self.alg_name == 'DPO':
                edited_model.unload()
                del self.model.peft_config
            elif self.alg_name == 'MELO':
                self.model = edited_model
            else:
                with torch.no_grad():
                    for k, v in weights_copy.items():
                        nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

            # for i, request in enumerate(record_chunks):
            #     chunk_metrics[i]["pre"] = compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device, test_generation=test_generation)

            #     if verbose:
            #         LOG.info(
            #             f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {chunk_metrics[i]}"
            #         )

            LOG.info(f"Evaluation took {time() - start}")
            all_metrics.extend(chunk_metrics)
        return all_metrics, edited_model, weights_copy





class SeracRewriteExecutor_new(SeracRewriteExecutor):
    def init_model(self, model, tok, params: SERACHparams):

        # assert params.archive is not None or print(f'Training weights Needed....')

        # Customize the gpt2xl and tokenizer
        self.model = model
        self.tokenizer = tok
        def set_padding():
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = 'left'
        set_padding()

        # # Load the trained MEND model
        # self.alg = SERAC(self.model, deepcopy(params), lambda: deepcopy(self.model))
        # d = torch.load(params.archive, map_location='cpu')
        # self.alg.load_state_dict(d["model"], False)
        # # self.alg.to(torch.device(f'cuda:{params.device}'))
        # self.alg.replacement.to(torch.device(f'cuda:{params.device}'))
        # self.alg.classifier.to(torch.device(f'cuda:{params.device}'))


        self.alg = SERAC(self.model, deepcopy(params), lambda: deepcopy(self.model))
        self.alg.classifier = getattr(transformers, params.cls_class).from_pretrained(params.cls_name).to(torch.device(f'cuda:{params.device}'))
        self.alg.classifier_tok = AutoTokenizer.from_pretrained(params.cls_name)
        self.alg.replacement = getattr(transformers, params.model_class).from_pretrained(params.small_name).to(torch.device(f'cuda:{params.device}'))
        self.alg.replacement_tok = AutoTokenizer.from_pretrained(params.small_name)
        self.alg.replacement_tok.pad_token_id = self.alg.replacement_tok.eos_token_id
        self.alg.replacement_tok.padding_side = 'left'

        self.is_init = True


