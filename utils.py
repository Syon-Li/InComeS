import torch
import math
import copy
import json
import random
import numpy as np
import itertools
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import word_tokenize, pos_tag, ne_chunk
import nltk
import string
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



def find_max_length(input_ids):
    max_length = 0
    for item in input_ids:
        if len(item) > max_length:
            max_length = len(item)
    return max_length



def recognize_names(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    named_entities = ne_chunk(pos_tags)
    # person_names = []
    rnt = False
    for chunk in named_entities:
        if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
            rnt = True
            break
            # person_names.append(" ".join([c[0] for c in chunk]))
    return rnt



def replace_what(prompt, target_new):
    pattern = r'(?i)\b(?:what|who|which|whom|when|where)\b'
    pattern_s = r"(?i)\b(?:what's|who's|which's)\b"
    pattern_au = r"(?i)\b(?:do|does|did)\b"
    # print(prompt, target_new)
    try:
        if len(re.findall(pattern_s, prompt))>0:
            edit = re.sub(pattern_s, "{}".format(target_new + " is"), prompt)
            edit = re.sub(pattern_au, "", edit)
            edit = edit.replace("?", "")
        elif len(re.findall(pattern, prompt))>0:
            edit = re.sub(pattern, "{}".format(target_new), prompt)
            edit = re.sub(pattern_au, "", edit)
            edit = edit.replace("?", "")
        else:
            edit = prompt + " " + target_new
    except:
        edit = prompt + " " + target_new
    return edit.strip()



def replace_subject(prompt, subject):
    pattern = r'(?i)\b{}\b'.format(re.escape(subject))
    pattern_s = r"(?i)\b{}'s\b".format(re.escape(subject))
    if len(re.findall(pattern_s, prompt))>0:
        edit = re.sub(pattern_s, "whose", prompt)
    else:
        edit = re.sub(pattern, "who or what", prompt)
    # edit = edit.replace(".", "?")
    edit += "?"
    return edit.strip()





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





def load_file(f_path, tokenizer):
    if "zsre" in f_path:
        with open(f_path) as f:
            ds = json.load(f)
        for record in ds:
            if record.get("alt",0)!=0:
                subject = record["subject"].strip()
                prompt = record["src"].strip()
                target_new = record["alt"].strip()
                m_edit = replace_what(prompt=prompt, target_new=target_new)
                m_rephrase_edit = replace_what(prompt=record["rephrase"].strip(), target_new=target_new)
                edit = m_edit
                rephrase_edit = m_rephrase_edit
                txt = prompt + " " + target_new
                yield tokenizer([edit, txt, prompt])
                # print(edit, txt, sep="\n")

                yield tokenizer([rephrase_edit, txt, prompt])

                locality_txt = record["loc"][len("nq question: "):].strip() + "?" + " " + record["loc_ans"].strip()
                locality_edit = replace_what(prompt=record["loc"][len("nq question: "):].strip(), 
                                            target_new=record["loc_ans"].strip())
                # locality_edit = locality_txt
                # print(locality_edit, locality_txt, sep="\n")
                yield tokenizer([locality_edit.capitalize(), 
                                locality_txt.capitalize(),
                                record["loc"][len("nq question: "):].strip() + "?"])                       

    elif "counterfact" in f_path:
        with open(f_path) as f:
            ds = json.load(f)
        # for i in range(2):
        for record in ds:
            subject = record["subject"].strip()
            prompt = record["prompt"].strip()
            target_new = record["target_new"].strip()
            edit = prompt + " " + target_new
            txt = prompt + " " + target_new
            yield tokenizer([edit, txt, prompt])      

            rephrase_edit = record["rephrase_prompt"].strip() + " " + record["target_new"].strip()
            yield tokenizer([rephrase_edit, txt, record["prompt"].strip()])

            locality_edit = record["locality_prompt"].strip() + " " + record["locality_ground_truth"].strip()
            yield tokenizer([locality_edit, 
                            locality_edit,
                            record["locality_prompt"].strip()])                 






def load_wiki(f_path, tokenizer=None):
    with open(f_path) as f:
        ds = json.load(f)    
                      
    for record in ds:
        subject = record["subject"].strip()
        edit = record["prompt"].strip() + " " + record["target_new"].strip()
        txt = record["prompt"].strip() + " " + record["target_new"].strip()
        yield tokenizer([edit, txt, record["prompt"].strip()], truncation=True)       

        if "portability" in record.keys():
            for port_key, port_value in record["portability"].items():
                for port_item in port_value:
                    if isinstance(port_item["ground_truth"], list):
                        for g_truth in port_item["ground_truth"]:
                            for g_truth_item in g_truth:
                                query = port_item["prompt"].strip() + " " + g_truth_item.strip()
                                yield tokenizer([query, query, port_item["prompt"].strip()], truncation=True)
                                yield tokenizer([edit, query, port_item["prompt"].strip()], truncation=True)
                    else:
                        query = port_item["prompt"].strip() + " " + port_item["ground_truth"].strip()
                        yield tokenizer([query, query, port_item["prompt"].strip()], truncation=True)
                        yield tokenizer([edit, query, port_item["prompt"].strip()], truncation=True)
                                        
        if "locality" in record.keys():
            for loc_key, loc_value in record["locality"].items():
                for loc_item in loc_value:
                    if isinstance(loc_item["ground_truth"], list):
                        for g_truth in loc_item["ground_truth"]:
                            for g_truth_item in g_truth:
                                query = loc_item["prompt"].strip() + " " + g_truth_item.strip()
                                q_txt = loc_item["prompt"].strip() + " " + g_truth_item.strip()
                                yield tokenizer([query, q_txt, loc_item["prompt"].strip()], truncation=True)

                            if len(g_truth)>=2:
                                for target_a, target_b in list(itertools.combinations(g_truth, 2)):
                                    contexts = loc_item["prompt"].strip() + " " + target_a.strip()
                                    query = loc_item["prompt"].strip() + " " + target_b.strip()
                                    yield tokenizer([contexts, query, loc_item["prompt"].strip()], truncation=True)
                    else:
                        query = loc_item["prompt"].strip() + " " + loc_item["ground_truth"].strip()
                        q_txt = loc_item["prompt"].strip() + " " + loc_item["ground_truth"].strip()
                        yield tokenizer([query, q_txt, loc_item["prompt"].strip()], truncation=True)









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

        def preprocess_s2orc(examples):
            examples["knowledge"] = examples["abstract"].strip()
            examples["txt"] = examples["title"].strip()
            return examples

        def preprocess_agnews(examples):
            examples["knowledge"] = examples["description"].strip()
            examples["txt"] = examples["title"].strip()
            return examples

        def preprocess_squad(examples):
            examples["knowledge"] =  examples["context"].strip()
            examples["txt"] = examples["question"].strip() + " " + examples["answers"]["text"][0].strip()
            examples["txt_stem"] = examples["question"].strip()
            return examples

        def preprocess_specter(examples):
            # examples["knowledge"] = "Write a sentence that is semantically similar to the following: " + examples["anchor"].strip()
            examples["knowledge"] = examples["anchor"].strip()
            examples["txt"] = examples["positive"].strip()
            return examples

        def preprocess_nq_simplified(examples):
            # examples["knowledge"] = "Write a sentence that is semantically similar to the following: " + examples["anchor"].strip()
            examples["knowledge"] = examples["context"].strip()
            examples["txt"] = examples["question"].strip() + " " + examples["answers"]["text"][0].strip()
            examples["txt_stem"] = examples["question"].strip()
            return examples

        def preprocess_openbookqa(examples):
            examples["knowledge"] = examples["fact1"].strip() if "fact1" in examples.keys() and isinstance(examples["fact1"], str) else ""
            answerKey = examples["answerKey"]
            label = examples["choices"]["label"]
            examples["txt"] = examples["question_stem"].strip() + " " + examples["choices"]["text"][label.index(answerKey)].strip()
            examples["txt_stem"] = examples["question_stem"].strip()
            return examples

        def preprocess_qasc(examples):
            examples["knowledge"] = examples["combinedfact"].strip() if "combinedfact" in examples.keys() and isinstance(examples["combinedfact"], str) else ""
            answerKey = examples["answerKey"]
            label = examples["choices"]["label"]
            examples["txt"] = examples["question"].strip() + " " + examples["choices"]["text"][label.index(answerKey)].strip()
            examples["txt_stem"] = examples["question"].strip()
            return examples

        def preprocess_medmcqa(examples):
            examples["knowledge"] = examples["exp"].strip() if "exp" in examples.keys() and isinstance(examples["exp"], str) else ""
            answerKey = examples["cop"]
            label = ["opa","opb","opc","opd"]
            examples["txt"] = examples["question"].strip() + " " + examples[label[answerKey]].strip()
            examples["txt_stem"] = examples["question"].strip()
            return examples

        def preprocess_exam(examples):
            examples["knowledge"] = examples["Explanation"].strip() if "Explanation" in examples.keys() and isinstance(examples["Explanation"], str) else ""
            if "Answer" in examples.keys() and examples["Answer"] is not None:
                # print(examples)
                # print(list(examples.keys()))
                answerKey = examples["Answer"]
                examples["txt"] = examples["Question"].strip() + " " + examples[answerKey].strip()
                examples["txt_stem"] = examples["Question"].strip()
            else:
                examples["txt"] = ""
            return examples

        def tokenize(examples):
            return self.tokenizer([examples["knowledge"].strip() if "knowledge" in examples.keys() else "", 
                                   examples["txt"].strip() if "txt" in examples.keys() else "",
                                   examples["txt_stem"].strip() if "txt_stem" in examples.keys() else ""],
                                   truncation=True)

        print(self.hf_f_path, self.f_path)

        
        iter_ds = []
        for path, subset, split in zip(self.hf_f_path, self.subset, self.split):
            ds = load_dataset(path, name=subset, split=split, streaming=True)
            # ds = load_dataset(path, name=subset, split=split)
            # ds = ds.to_iterable_dataset()
            if "s2orc" in path.lower():
                ds = ds.map(preprocess_s2orc)
            elif "agnews" in path.lower():
                ds = ds.map(preprocess_agnews)
            elif "squad" in path.lower():
                ds = ds.map(preprocess_squad)
            elif "specter" in path.lower():
                ds = ds.map(preprocess_specter)
            elif "nq-simplified" in path.lower():
                ds = ds.map(preprocess_nq_simplified)
            elif "openbookqa" in path:
                ds = ds.map(preprocess_openbookqa)
            elif "allenai/qasc" in path:
                ds = ds.map(preprocess_qasc)
            elif "medmcqa" in path:
                ds = ds.map(preprocess_medmcqa)
            elif "neteval-exam" in path:
                ds = ds.map(preprocess_exam)
            ds = ds.map(tokenize)
            iter_ds.append(iter(ds))
        
        for path in self.f_path:
            if "zsre" in path.lower() or "counterfact-train" in path.lower():
                iter_ds.append(load_file(path, tokenizer=self.tokenizer))
            else:
                iter_ds.append(load_wiki(path, tokenizer=self.tokenizer))
        
        lower_bound = 2 if self.tokenizer.bos_token_id is not None and self.tokenizer.bos_token_id != self.tokenizer.pad_token_id else 1

        s2orc_idx = self.hf_f_path.index("sentence-transformers/s2orc")

        f_idxs = list(range(len(self.hf_f_path)+len(self.f_path)))
        # f_idxs = list(range(len(self.f_path)))
        # f_idxs = list(range(len(self.hf_f_path)))

        cnts = [0 for _ in range(len(self.hf_f_path)+len(self.f_path))]
        while(len(f_idxs)>0):
            idx = random.choices(f_idxs, weights=self.weights, k=1)[0]
            while True:
                try:
                    inputs = next(iter_ds[idx])
                    kn, txt, txt_stem = inputs["input_ids"][0], inputs["input_ids"][1], inputs["input_ids"][2]
                    if len(kn) > lower_bound and len(kn) <= 128 and len(txt) > lower_bound:
                        kn.append(self.gist_token_ids)
                        kn_txt = copy.deepcopy(kn)
                        kn_txt.append(self.tokenizer.encode("\n")[-1])
                        if lower_bound==2:
                            kn_txt.extend(txt[1:])
                        else:
                            kn_txt.extend(txt)
                        if len(kn_txt) <= 256:
                            # print(kn, txt, kn_txt)
                            cnts[idx] += 1
                            yield (kn, txt, kn_txt, txt_stem)
                            # yield (kn, txt[:128], kn_txt[:len(kn)+128-1])
                            break
                except StopIteration:
                    self.weights.pop(f_idxs.index(idx))
                    f_idxs.remove(idx)
                    break

            if s2orc_idx in f_idxs and cnts[s2orc_idx] == 4.5e06:
                break    

            if self.lines is not None:
                if sum(cnts) == self.lines:
                    break 




def padding_fn(chunk, pad_id, gist_token_ids, txt_stem_chunk=None, bos_token_id=None):
    input_ids, attention_masks, labels, T_labels = [], [], [], []
    max_length = find_max_length(chunk)
    for i, data in enumerate(chunk):
        if gist_token_ids in data:
            gist_loc = data.index(gist_token_ids)
        else:
            gist_loc = -1
        if txt_stem_chunk is not None:
            if len(txt_stem_chunk[i])>0 and bos_token_id is not None and txt_stem_chunk[i][-1]!=bos_token_id:
                gist_loc += len(txt_stem_chunk[i])
            else:
                gist_loc += min(math.floor(len(data)*0.5), 12)
        difference = max_length - len(data)
        attention_mask = torch.zeros(max_length)
        attention_mask[:len(data)] = 1
        T_label = data + [-100] * difference
        label = [-100] * (gist_loc + 1) + data[(gist_loc + 1):] + [-100] * difference
        data = data + [pad_id] * difference
        input_ids.append(data)
        attention_masks.append(attention_mask.tolist())
        labels.append(label)
        T_labels.append(T_label)
    return torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(labels), torch.tensor(T_labels)




def wrap_collate_fn(pad_id, gist_token_ids):
    def collate_fn_fileD(batch, pad_id=pad_id, gist_token_ids=gist_token_ids):
        # print(batch, len(batch))
        # print(batch[0])
        kns, txts, kn_txts = [], [], []
        for inputs in batch:
            kn, txt, kn_txt = inputs
            kns.append(kn)
            txts.append(txt)
            kn_txts.append(kn_txt)

        input_ids_kn, attention_mask_kn, _ = padding_fn(kns, pad_id, gist_token_ids)
        input_ids_txt, attention_mask_txt, labels = padding_fn(txts, pad_id, gist_token_ids)
        input_ids, attention_mask, _ = padding_fn(kn_txts, pad_id, gist_token_ids)

        return input_ids_kn, attention_mask_kn, input_ids_txt, attention_mask_txt, input_ids, attention_mask, labels
    return collate_fn_fileD




def collate_fn(batch):
    # print(batch, len(batch))
    # print(batch[0])
    kns, txts, kn_txts, txt_stems = [], [], [], []
    for inputs in batch:
        kn, txt, kn_txt, txt_stem = inputs
        kns.append(kn)
        txts.append(txt)
        kn_txts.append(kn_txt)
        txt_stems.append(txt_stem)
    return kns, txts, kn_txts, txt_stems





def mask_gist(input_ids, attention_mask, gist_token_ids, pad_id):
    bsz, q_len = input_ids.shape
    gist_loc_vec = (input_ids == gist_token_ids).logical_not()
    input_ids = input_ids.clone()
    attention_mask = attention_mask.clone()

    input_ids_new = input_ids[gist_loc_vec].reshape(bsz, q_len-1)
    attention_mask_new = attention_mask[gist_loc_vec].reshape(bsz, q_len-1)

    input_ids_new = torch.concat([torch.ones(bsz, 1, dtype=input_ids.dtype, device=input_ids.device)*pad_id, input_ids_new], dim=-1)
    attention_mask_new = torch.concat([torch.zeros(bsz, 1, dtype=attention_mask.dtype, device=attention_mask.device), attention_mask_new], dim=-1)
    
    return input_ids_new, attention_mask_new




def mask_context(input_ids, attention_mask, gist_token_ids, pad_id, bos_tok_id):
    gist_loc_vec = (input_ids == gist_token_ids)
    input_ids = input_ids.clone()
    attention_mask = attention_mask.clone()
    before_gist_mask = (reverse_cumsum(gist_loc_vec) > 0)
    input_ids[before_gist_mask] = pad_id
    attention_mask[before_gist_mask] = 0
    # add bos token
    if bos_tok_id is not None:
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

        


if __name__=="__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_path = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, cache_dir="/apdcephfs_qy3/share_733425/zhisonzhang/users/shuaiyili/transformers_cache")
    tokenizer.pad_token = tokenizer.eos_token

    # f_path = ["./KnowEdit/benchmark/wiki_counterfact/train_cf.json", "./KnowEdit/benchmark/wiki_recent/recent_train.json"]
    # for i, dta in enumerate(load_wiki(f_path[0], tokenizer=tokenizer)):
    #     pass
    # print(i+1)
    # # print(tokenizer.batch_decode(dta["input_ids"]))


    f_path = ["./Editing_data/zsre/zsre_mend_train.json", "./Editing_data/counterfact/counterfact-train.json"]
    for i, dta in enumerate(load_file(f_path[0], tokenizer=tokenizer)):
        pass
    print(i+1)

