import torch
import math
import copy
import json
import random
from typing import Tuple, Iterator, List
import numpy as np
import itertools
import ast
from nltk.tokenize import sent_tokenize, word_tokenize
import torch.nn.functional as F
# from liger_kernel.transformers.kl_div import LigerKLDIVLoss
from datasets import load_dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



# learning rate decay scheduler (cosine with warmup)
def get_lr(it, warmup_iters, lr_decay_iters, min_lr, learning_rate):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)




def get_num_lines(f_path:str):
    with open(f_path, "r") as f:
        num_lines = 1
        for line in f:
            if line.strip():
                num_lines += 1
    return num_lines




def reverse_cumsum(x: torch.Tensor) -> torch.Tensor:
    """Cumulative sum from right to left.
    See https://github.com/pytorch/pytorch/issues/33520.
    """
    return x + torch.sum(x, dim=-1, keepdims=True) - torch.cumsum(x, dim=-1)


def make_mask_pre_first_gist(inputs: torch.Tensor, gist_token: int, dtype=torch.int64) -> torch.Tensor:
    """Returns a mask where all tokens prior to the first gist token are masked out.

    Args:
        inputs: a Tensor of input tokens where the last dimension is the sequence length.
        gist_token: the integer id of the gist token.
        dtype: the dtype of the mask, default int64.
    Returns:
        The requested mask."""
    return ((inputs == gist_token).cumsum(-1) >= 1).type(dtype)


def make_mask_post_last_gist(inputs: torch.Tensor, gist_token: int, dtype=torch.int64) -> torch.Tensor:
    """Returns a mask where all tokens after the last gist token are masked out.

    Computes the same as mask_pre_first_gist_token, but reverses the sequence before and after the cumsum.

    Args:
        inputs: a Tensor of input tokens where the last dimension is the sequence length.
        gist_token: the integer id of the gist token.
        dtype: the dtype of the mask, default int64.
    Returns:
        The requested mask.
    """
    return (reverse_cumsum(inputs == gist_token) >= 1).type(dtype)


def make_gist_mask(inputs: torch.Tensor, gist_token: int, pad_token: int, dtype=torch.int64) -> torch.Tensor:
    """Creates a gist mask from supplied inputs and gist/pad tokens.

    Tokens after the last gist cannot attend to tokens prior to the first gist. Additionally, tokens *before*
    the last gist cannot attend to tokens *after* the last gist.

    The gist mask is broadcasted to 4D (with a singleton dim 1) for compatibility with multi-headed attention
    (where dim 1 is the head dimension).

    Args:
        inputs: a Tensor of shape (batch_size, seq_len) input tokens.
        gist_token: the integer id of the gist token.
        pad_token: the integer id of the pad token. inputs == pad_token are masked out.
        dtype: the dtype of the mask, default int64.
    Returns:
        The requested mask of shape (batch_size, 1, seq_len, seq_len)
    """
    # Attention mask for tokens before the last gist token.
    pre_gist_mask = make_mask_post_last_gist(inputs, gist_token, dtype=torch.bool)[:, None, None]
    # Attention mask for tokens after the last gist token.
    post_gist_mask = make_mask_pre_first_gist(inputs, gist_token, dtype=torch.bool)[:, None, None]
    # Construct time masks by permuting to time dimension.
    pre_gist_time_mask = pre_gist_mask.permute((0, 1, 3, 2))

    mask = torch.where(pre_gist_time_mask, pre_gist_mask, post_gist_mask)
    mask = mask & (inputs != pad_token)[:, None, None] # Mask out pad tokens.

    return mask.type(dtype)



class ArrIterD(torch.utils.data.IterableDataset):
    def __init__(self, mem_arr, gist_token_ids, gist_location_id, config):
        super(ArrIterD).__init__()
        self.arr = mem_arr
        self.input_ids = chunk_arr(mem_arr)
        self.start = 0
        self.end = len(self.input_ids)
        self.gist_token_ids = gist_token_ids
        self.gist_location_id = gist_location_id
        self.config = config

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            print("worker id:", worker_id)
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

            input_ids, attention_masks, labels = [], [], []
            max_length = find_max_length(self.input_ids[iter_start:iter_end]) + 1 # add one extra gist token
            for input_id in self.input_ids[iter_start:iter_end]:
                gist_loc = input_id.index(self.gist_location_id)
                input_id.insert(gist_loc, self.gist_token_ids)
                label = input_id.copy()
                attention_mask = torch.zeros(max_length)
                attention_mask[:len(input_id)] = 1
                difference = max_length - len(input_id)
                label = [-100] * (gist_loc + 1) + input_id[(gist_loc + 1):] + [-100] * difference
                input_id = input_id + [self.config.eos_token_id] * difference
                input_ids.append(torch.tensor(input_id))
                attention_masks.append(attention_mask)
                labels.append(torch.tensor(label))

        dataset = tuple(zip(input_ids, attention_masks, labels))
        return iter(dataset)
        # return iter(self.chunk_list[iter_start:iter_end])
    
    def __len__(self):
        return (self.end - self.start)
    





class FileIterD(torch.utils.data.IterableDataset):
    def __init__(self, f_path_set, start_set, end_set, batch_size):
        super(FileIterD).__init__()
        self.f_path_set = f_path_set
        self.start_set = start_set
        self.end_set = end_set
        self.batch_size = batch_size
    
    def __len__(self):
        leg = 0
        for start, end in zip(self.start_set, self.end_set):
            leg += (end - start)
        return leg

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # print(worker_info)
        random.seed(126)

        iter_start_set = self.start_set

        iter_end_set = []
        for r in self.end_set:
            if r == -1:
                iter_end_set.append(float("inf"))
            else:
                iter_end_set.append(r)

        # if worker_info is not None:
        #     iter_start_set, iter_end_set = [], []
        #     for start, end in zip(self.start_set, self.end_set):
        #         per_worker = int(math.ceil((end-start) / float(worker_info.num_workers)))
        #         worker_id = worker_info.id
        #         iter_start = worker_id * per_worker
        #         iter_end = min(start + per_worker, end)
        #         iter_start_set.append(iter_start)
        #         iter_end_set.append(iter_end)
        
        f_objects = []
        for i,f_path in enumerate(self.f_path_set):
            f_objects.append(open(f_path, "r"))

        cnts = [0] * len(self.f_path_set)
        for i,f in enumerate(f_objects):
            for _ in range(iter_start_set[i]):
                f.readline()
                cnts[i] += 1
        
        f_idxs = list(range(len(self.f_path_set)))
        weights = [2.2, 4.1, 9.3, 15]
        while len(f_idxs)>0:
            idx = random.choices(f_idxs, weights=weights, k=1)[0]
            batch_r = min(cnts[idx]+self.batch_size[idx], iter_end_set[idx])
            input_ids = []
            cnt = 0
            for _ in range(cnts[idx], batch_r):
                line = f_objects[idx].readline()
                if line.strip():
                    py_dict = json.loads(line.strip())
                    input_ids.append(py_dict["input_ids"])  
                    cnt += 1              
                else:
                    weights.pop(f_idxs.index(idx))
                    f_idxs.remove(idx)
                    break
            cnts[idx] += cnt
            if cnts[idx] >= iter_end_set[idx] - 1:
                weights.pop(f_idxs.index(idx))
                f_idxs.remove(idx)

            if len(input_ids)>0:
                yield input_ids

        for f in f_objects:
            f.close()  
                





def chunk_arr(mem_arr:np.array):
    mem_arr_c = np.copy(mem_arr)
    eos_idx = np.argwhere(mem_arr_c==2).squeeze()
    ds = []
    ds.append(mem_arr_c[:eos_idx[0]+1].tolist())
    for i in range(1, len(eos_idx)):
        input_id = mem_arr_c[eos_idx[i-1]+1:eos_idx[i]+1].tolist()
        ds.append(input_id)
    return ds


def find_max_length(input_ids):
    max_length = 0
    for item in input_ids:
        if len(item) > max_length:
            max_length = len(item)
    return max_length





# def wrap_collate_fn(pad_id, gist_token_ids):
#     def collate_fn_fileD(batch, pad_id=pad_id, gist_token_ids=gist_token_ids):
#         # print(batch, len(batch))
#         # print(batch[0])
#         input_ids, attention_masks, labels = [], [], []
#         max_length = find_max_length(batch)
#         for data in batch:
#             if gist_token_ids in data:
#                 gist_loc = data.index(gist_token_ids)
#                 # gist_loc += 1
#                 # gist_loc = 0
#                 # for i, x in enumerate(reversed(data)): 
#                 #     if x == gist_token_ids:
#                 #         gist_loc = i
#                 #         break
#             else:
#                 gist_loc = -1
#             # gist_loc = data.index(gist_token_ids)
#             difference = max_length - len(data)
#             attention_mask = torch.zeros(max_length)
#             attention_mask[:len(data)] = 1
#             label = [-100] * (gist_loc + 1) + data[(gist_loc + 1):] + [-100] * difference
#             data = data + [pad_id] * difference
#             input_ids.append(data)
#             attention_masks.append(attention_mask.tolist())
#             labels.append(label)

#         return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_masks), torch.tensor(labels)
#     return collate_fn_fileD







def alter_position_ids(gist_token_ids: int, input_ids: torch.Tensor, origin_pos_ids: torch.tensor):
    # print(input_ids, gist_token_ids)
    # print(origin_pos_ids)
    gist_loc = torch.nonzero(input_ids == gist_token_ids)
    if len(gist_loc)>0:
        # pos_ids = origin_pos_ids.repeat_interleave(input_ids.shape[0], dim=0) - (gist_loc[1::2,-1] + 1).reshape(input_ids.shape[0], -1)
        pos_ids = origin_pos_ids.repeat_interleave(input_ids.shape[0], dim=0) - (gist_loc[...,-1]).reshape(input_ids.shape[0], -1)
    else:
        pos_ids = origin_pos_ids
    return pos_ids
  





# def load_file(f_path, tokenizer):
#     if "zsre" in f_path:
#         with open(f_path) as f:
#             ds = json.load(f)
#         for record in ds:
#             if record.get("alt",0)!=0:
#                 prompt = record["src"]
#                 target_new = record["alt"]
#                 edit = prompt.strip() + " " + target_new.strip()
#                 txts = []
#                 txts.append(edit + " <GIST> " + edit)
#                 txts.append(edit + " <GIST> " + record["rephrase"].strip() + " " + target_new.strip())
#                 txts.append(edit + " <GIST> " + record["loc"][len("nq question: "):].strip() + " " + record["loc_ans"].strip())
#                 for txt in txts:            
#                     yield tokenizer(txt, truncation=True)
#     elif "counterfact" in f_path:
#         with open(f_path) as f:
#             ds = json.load(f)
#         for record in ds:
#             edit = record["prompt"].strip() + " " + record["target_new"].strip()
#             txts = []
#             txts.append(edit + " <GIST> " + edit)
#             txts.append(edit + " <GIST> " + record["rephrase_prompt"].strip() + " " + record["target_new"].strip())
#             txts.append(edit + " <GIST> " + record["locality_prompt"].strip() + " " + record["locality_ground_truth"].strip())
#             for txt in txts:
#                 yield tokenizer(txt, truncation=True)
#     else:
#         cnt = 0
#         with open(f_path) as f:
#             while True:
#                 line = f.readline()
#                 if line.strip() and cnt<512000:
#                     py_dict = ast.literal_eval(line.strip())
#                     # print(py_dict)
#                     # print(type(py_dict["input_ids"][0]))
#                     yield py_dict
#                     cnt += 1
#                 else:
#                     break        



        


# class HFIterD(torch.utils.data.IterableDataset):
#     def __init__(self, hf_f_path, f_path, chunk_size, subset, split, tokenizer, weights, lines):
#         super(HFIterD).__init__()
#         self.hf_f_path = hf_f_path
#         self.f_path = f_path
#         self.subset = subset if subset else None
#         self.split = split if split else None
#         self.tokenizer = tokenizer
#         self.gist_token_ids = tokenizer.additional_special_tokens_ids[-1]
#         self.weights = weights
#         self.lines = lines if lines else None

#     def __iter__(self):
#         random.seed(126)       
        
#         def preprocess_para(examples):
#             examples["input_txt"] = examples["short"] + " <GIST> " + examples["long"]
#             return examples
            
#         def preprocess_nli(examples):
#             examples["input_txt"] = examples["anchor"] + " <GIST> " + examples["positive"]
#             return examples

#         def preprocess_lamini(examples):
#             examples["input_txt"] = examples["instruction"] + " <GIST> " + examples["response"]
#             return examples

#         def preprocess_competition(examples):
#             examples["input_txt"] = examples["problem"] + " <GIST> " + examples["solution"]
#             return examples

#         def preprocess_gsm8k(examples):
#             examples["input_txt"] = examples["question"] + " <GIST> " + examples["answer"]
#             return examples

#         def preprocess_math_hard(examples):
#             examples["input_txt"] = examples["problem"] + " <GIST> " + examples["solution"]
#             return examples  

#         def tokenize(examples):
#             return self.tokenizer(examples["input_txt"], truncation=True) 

#         iter_ds = []
#         for path, subset, split in zip(self.hf_f_path, self.subset, self.split):
#             ds = load_dataset(path, name=subset, split=split, streaming=True)
#             if "paranmt5m" in path:
#                 ds = ds.map(preprocess_para)
#             elif "all-nli" in path:
#                 ds = ds.map(preprocess_nli)
#             elif "LaMini" in path:
#                 ds = ds.map(preprocess_lamini)
#             elif "competition" in path:
#                 ds = ds.map(preprocess_competition)
#             elif "gsm8k" in path:
#                 ds = ds.map(preprocess_gsm8k)
#             elif "MATH-Hard" in path:
#                 ds = ds.map(preprocess_math_hard)
#             ds = ds.map(tokenize)
#             iter_ds.append(iter(ds))
        
#         for path in self.f_path:
#             iter_ds.append(load_file(path, tokenizer=self.tokenizer))
            

#         f_idxs = list(range(len(self.hf_f_path)+len(self.f_path)))
#         cnt = 0
#         while(len(f_idxs)>0):
#             idx = random.choices(f_idxs, weights=self.weights, k=1)[0]
#             try:
#                 inputs = next(iter_ds[idx])
#                 if len(inputs["input_ids"])<=256:
#                     yield inputs["input_ids"]
#                 else:
#                     if inputs["input_ids"].index(self.gist_token_ids)+1 < 128:
#                         yield inputs["input_ids"][:256]            
#             except StopIteration:
#                 self.weights.pop(f_idxs.index(idx))
#                 f_idxs.remove(idx)
#             cnt += 1
#             if self.lines:
#                 if cnt == self.lines:
#                     break  




def load_file(f_path, tokenizer):
    if "zsre" in f_path:
        with open(f_path) as f:
            ds = json.load(f)
        # streaming output each instances, avoiding same edits in the same gist batch
        for i in range(4):
            for record in ds:
                if record.get("alt",0)!=0:
                    prompt = record["src"]
                    target_new = record["alt"]
                    edit = prompt.strip() + " " + target_new.strip()
                    if i==0:
                        inputs = [edit, 
                                  edit,]
                    elif i==1:
                        rephrase_edit = record["rephrase"].strip() + " " + target_new.strip()
                        inputs = [rephrase_edit, 
                                  edit,]
                    elif i==2:
                        locality_edit = record["loc"][len("nq question: "):].strip() + " " + record["loc_ans"].strip()
                        inputs = [locality_edit, 
                                  locality_edit,]
                    # elif i==3:
                    #     locality_edit = record["loc"][len("nq question: "):].strip() + " " + record["loc_ans"].strip()
                    #     inputs = [edit, 
                    #               locality_edit,]                          
                    yield tokenizer(inputs, truncation=True)
    elif "counterfact" in f_path:
        with open(f_path) as f:
            ds = json.load(f)
        for i in range(4):
            for record in ds:
                edit = record["prompt"].strip() + " " + record["target_new"].strip()
                if i==0:
                    inputs = [edit, 
                              edit,]
                elif i==1:
                    rephrase_edit = record["rephrase_prompt"].strip() + " " + record["target_new"].strip()
                    inputs = [rephrase_edit, 
                              edit,]
                elif i==2:
                    locality_edit = record["locality_prompt"].strip() + " " + record["locality_ground_truth"].strip()
                    inputs = [locality_edit, 
                              locality_edit,]
                # elif i==3:
                #     locality_edit = record["locality_prompt"].strip() + " " + record["locality_ground_truth"].strip()
                #     inputs = [edit, 
                #               locality_edit,]                    
                yield tokenizer(inputs, truncation=True)





class HFIterD(torch.utils.data.IterableDataset):
    def __init__(self, hf_f_path, f_path, subset, split, tokenizer, weights, lines):
        super(HFIterD).__init__()
        self.hf_f_path = hf_f_path
        self.f_path = f_path
        self.subset = subset if subset else None
        self.split = split if split else None
        self.tokenizer = tokenizer
        self.gist_token_ids = tokenizer.additional_special_tokens_ids[-1]
        self.weights = weights
        self.lines = lines if lines else None

    def __iter__(self):
        
        def preprocess_para(examples):
            examples["knowledge"] = examples["short"]
            examples["txt"] = examples["long"]
            return examples
            
        def preprocess_nli(examples):
            examples["knowledge"] = examples["anchor"]
            examples["txt"] = examples["positive"]
            return examples

        def preprocess_lamini(examples):
            examples["knowledge"] = examples["instruction"]
            examples["txt"] = examples["response"]
            return examples

        def preprocess_wiki_paraphrased(examples):
            examples["knowledge"] = examples["original"]
            examples["txt"] = examples["paraphrase"]
            return examples

        def preprocess_openbookqa(examples):
            examples["knowledge"] = examples["fact1"] if "fact1" in examples.keys() and isinstance(examples["fact1"], str) else ""
            answerKey = examples["answerKey"]
            label = examples["choices"]["label"]
            examples["txt"] = examples["question_stem"].strip() + " " + examples["choices"]["text"][label.index(answerKey)].strip()
            return examples

        def preprocess_qasc(examples):
            examples["knowledge"] = examples["combinedfact"] if "combinedfact" in examples.keys() and isinstance(examples["combinedfact"], str) else ""
            answerKey = examples["answerKey"]
            label = examples["choices"]["label"]
            examples["txt"] = examples["question"].strip() + " " + examples["choices"]["text"][label.index(answerKey)].strip()
            return examples

        def preprocess_medmcqa(examples):
            examples["knowledge"] = examples["exp"] if "exp" in examples.keys() and isinstance(examples["exp"], str) else ""
            answerKey = examples["cop"]
            label = ["opa","opb","opc","opd"]
            examples["txt"] = examples["question"].strip() + " " + examples[label[answerKey]].strip()
            return examples

        def preprocess_exam(examples):
            examples["knowledge"] = examples["Explanation"] if "Explanation" in examples.keys() and isinstance(examples["Explanation"], str) else ""
            if "Answer" in examples.keys() and examples["Answer"] is not None:
                # print(examples)
                # print(list(examples.keys()))
                answerKey = examples["Answer"]
                examples["txt"] = examples["Question"].strip() + " " + examples[answerKey].strip()
            else:
                examples["txt"] = ""
            return examples

        def preprocess_gsm8k(examples):
            q_sents = sent_tokenize(examples["question"])
            k = ""
            if len(q_sents) > 1:
                for i,sent in enumerate(q_sents):
                    if i<len(q_sents)-1:
                        k += sent.strip() + " "
                examples["knowledge"] = k.strip()
                examples["txt"] = q_sents[-1].strip() + " " + examples["answer"].strip()
            else:
                examples["knowledge"] = q_sents[-1].strip()
                examples["txt"] = examples["answer"].strip()
            return examples 

        def tokenize(examples):
            # if isinstance(examples["knowledge"], str) and isinstance(examples["txt"], str):
            #     return self.tokenizer([examples["knowledge"], examples["txt"]], truncation=True)
            # else:
            #     return {"input_ids":[[-1], [-1]]}
            return self.tokenizer([examples["knowledge"] if "knowledge" in examples.keys() else "", 
                                   examples["txt"] if "txt" in examples.keys() else ""], truncation=True)

        print(self.hf_f_path, self.f_path)

        
        iter_ds = []
        for path, subset, split in zip(self.hf_f_path, self.subset, self.split):
            # ds = load_dataset(path, name=subset, split=split, streaming=True)
            ds = load_dataset(path, name=subset, split=split)
            ds = ds.to_iterable_dataset()
            #["ltg/en-wiki-paraphrased", "MBZUAI/LaMini-instruction", "sentence-transformers/all-nli", "allenai/openbookqa", "allenai/qasc", "openlifescienceai/medmcqa", "NASP/neteval-exam"]
            if "all-nli" in path:
                ds = ds.map(preprocess_nli)
            # elif "paranmt5m" in path:
            #     ds = ds.map(preprocess_para)
            elif "en-wiki-paraphrased" in path:
                ds = ds.map(preprocess_wiki_paraphrased)
            elif "LaMini-instruction" in path:
                ds = ds.map(preprocess_lamini)
            elif "openbookqa" in path:
                ds = ds.map(preprocess_openbookqa)
            # elif "preprocess_qasc" in path:
            #     ds = ds.map(preprocess_qasc)
            elif "allenai/qasc" in path:
                ds = ds.map(preprocess_qasc)
            elif "medmcqa" in path:
                ds = ds.map(preprocess_medmcqa)
            elif "neteval-exam" in path:
                ds = ds.map(preprocess_exam)
            # elif "gsm8k" in path:
            #     ds = ds.map(preprocess_gsm8k)
            ds = ds.map(tokenize)
            iter_ds.append(iter(ds))
        
        for path in self.f_path:
            iter_ds.append(load_file(path, tokenizer=self.tokenizer))
            
        lamini_idx = self.hf_f_path.index("MBZUAI/LaMini-instruction")
        wiki_paraphrased_idx = self.hf_f_path.index("ltg/en-wiki-paraphrased")
        f_idxs = list(range(len(self.hf_f_path)+len(self.f_path)))
        # f_idxs = list(range(len(self.f_path)))
        cnts = [0 for _ in range(len(self.hf_f_path)+len(self.f_path))]
        while(len(f_idxs)>0):
            idx = random.choices(f_idxs, weights=self.weights, k=1)[0]
            # print("selected file idx", idx)
            try:
                inputs = next(iter_ds[idx])
                kn, txt = inputs["input_ids"][0], inputs["input_ids"][1]
                if len(kn) > 1 and len(kn) < 128 and len(txt) > 1:
                    kn.append(self.gist_token_ids)
                    kn_txt = copy.deepcopy(kn)
                    kn_txt.extend(txt[1:])
                    if len(kn_txt) <= 256:
                        # print(kn, txt, kn_txt)
                        # yield (kn, txt, kn_txt)
                        yield (kn, txt[:128], kn_txt[:len(kn)+128-1])
            except StopIteration:
                self.weights.pop(f_idxs.index(idx))
                f_idxs.remove(idx)
                # if idx == wiki_paraphrased_idx:
                #     self.weights.clear()
                #     f_idxs.clear()   

            cnts[idx] += 1
            if lamini_idx in f_idxs and cnts[lamini_idx] == 1.5e06:
                self.weights.pop(f_idxs.index(lamini_idx))
                f_idxs.remove(lamini_idx)    

            if wiki_paraphrased_idx in f_idxs and cnts[wiki_paraphrased_idx] == 5e06:
                # self.weights.pop(f_idxs.index(wiki_paraphrased_idx))
                # f_idxs.remove(wiki_paraphrased_idx)  
                self.weights.clear()
                f_idxs.clear()     

            # print("cnts", cnts)       

            if self.lines is not None:
                if sum(cnts) == self.lines:
                    break 



def padding_fn(chunk, pad_id, gist_token_ids):
    input_ids, attention_masks, labels = [], [], []
    max_length = find_max_length(chunk)
    for data in chunk:
        if gist_token_ids in data:
            gist_loc = data.index(gist_token_ids)
        else:
            gist_loc = -1
        difference = max_length - len(data)
        attention_mask = torch.zeros(max_length)
        attention_mask[:len(data)] = 1
        label = [-100] * (gist_loc + 1) + data[(gist_loc + 1):] + [-100] * difference
        data = data + [pad_id] * difference
        input_ids.append(data)
        attention_masks.append(attention_mask.tolist())
        labels.append(label)
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_masks), torch.tensor(labels)




def wrap_collate_fn(pad_id, gist_token_ids):
    def collate_fn_fileD(batch, pad_id=pad_id, gist_token_ids=gist_token_ids):
        # print(batch, len(batch))
        # print(batch[0])
        batch_aug = ([], [], [])
        for inputs in batch:
            kn, txt, kn_txt = inputs
            batch_aug[0].append(kn)
            batch_aug[1].append(txt)
            batch_aug[2].append(kn_txt)

        input_ids_kn, attention_mask_kn, _ = padding_fn(batch_aug[0], pad_id, gist_token_ids)
        input_ids_txt, attention_mask_txt, _ = padding_fn(batch_aug[1], pad_id, gist_token_ids)
        input_ids, attention_mask, labels = padding_fn(batch_aug[2], pad_id, gist_token_ids)

        return input_ids_kn, attention_mask_kn, input_ids_txt, attention_mask_txt, input_ids, attention_mask, labels
    return collate_fn_fileD




def collate_fn(batch):
    # print(batch, len(batch))
    # print(batch[0])
    kns, txts, kn_txts = [], [], []
    for inputs in batch:
        kn, txt, kn_txt = inputs
        kns.append(kn)
        txts.append(txt)
        kn_txts.append(kn_txt)
    return kns, txts, kn_txts





def remove_gist(input_ids, attention_mask, gist_token_ids, pad_id):
    bsz, q_len = input_ids.shape
    gist_loc_vec = (input_ids == gist_token_ids).logical_not()
    input_ids = input_ids.clone()
    attention_mask = attention_mask.clone()

    input_ids_new = input_ids[gist_loc_vec].reshape(bsz, q_len-1)
    attention_mask_new = attention_mask[gist_loc_vec].reshape(bsz, q_len-1)

    input_ids_new = torch.concat([torch.ones(bsz, 1, dtype=input_ids.dtype, device=input_ids.device)*pad_id, input_ids_new], dim=-1)
    attention_mask_new = torch.concat([torch.zeros(bsz, 1, dtype=attention_mask.dtype, device=attention_mask.device), attention_mask_new], dim=-1)
    
    return input_ids_new, attention_mask_new



def remove_context(input_ids, attention_mask, gist_token_ids, pad_id, bos_tok_id):
    gist_loc_vec = (input_ids == gist_token_ids)
    input_ids = input_ids.clone()
    attention_mask = attention_mask.clone()
    before_gist_mask = (reverse_cumsum(gist_loc_vec) > 0)
    input_ids[before_gist_mask] = pad_id
    attention_mask[before_gist_mask] = 0
    # add bos token
    input_ids[gist_loc_vec] = bos_tok_id
    attention_mask[gist_loc_vec] = 1
    # print(input_ids, input_ids.shape)
    
    return input_ids, attention_mask





def _chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    if isinstance(arr, (list, torch.Tensor)):
        for i in range(0, len(arr), n):
            yield arr[i: i + n]
    elif isinstance(arr, dict):
        field = list(arr.keys())
        for i in range(0, len(arr[field[0]]["prompt"]), n):
            rnt = {}
            for k,v in arr.items():
                rnt.update({k:{"prompt":v["prompt"][i: i + n], "ground_truth":v["ground_truth"][i: i + n]}})
            yield rnt




# get a derangement
def sattolo_cycle(n):
    permutation = list(range(n))
    i = n - 1
    while i > 0:
        # 选择一个比 i 小的随机索引
        j = random.randint(0, i - 1)
        # 交换 i 和 j 位置的元素
        permutation[i], permutation[j] = permutation[j], permutation[i]
        i -= 1
    return permutation





