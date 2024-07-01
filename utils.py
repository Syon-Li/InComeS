import torch
import math
import copy
import json
import random
from typing import Tuple, Iterator, List
import numpy as np


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
        iter_end_set = self.end_set
        if worker_info is not None:
            iter_start_set, iter_end_set = [], []
            for start, end in zip(self.start_set, self.end_set):
                per_worker = int(math.ceil((end-start) / float(worker_info.num_workers)))
                worker_id = worker_info.id
                iter_start = worker_id * per_worker
                iter_end = min(start + per_worker, end)
                iter_start_set.append(iter_start)
                iter_end_set.append(iter_end)
        
        f_objects = []
        for i,f_path in enumerate(self.f_path_set):
            f_objects.append(open(f_path, "r"))

        cnts = [0] * len(self.f_path_set)
        for i,f in enumerate(f_objects):
            for _ in range(iter_start_set[i]):
                f.readline()
                cnts[i] += 1
        
        f_idxs = list(range(len(self.f_path_set)))
        batch = 0
        while len(f_idxs)>0:
            idx = random.sample(f_idxs,k=1)[0]
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
                    f_idxs.remove(idx)
                    break
            cnts[idx] += cnt
            if cnts[idx]+1 >= iter_end_set[idx]:
                f_idxs.remove(idx)

            yield batch,input_ids
            batch += 1

        for f in f_objects:
            f.close()  
                




class CustomBatchSampler(torch.utils.data.Sampler):
    def __init__(self, chaindataset: torch.utils.data.ChainDataset, batch_size: List[int]):
        self.chaindataset = chaindataset
        self.batch_size = batch_size
    
    def __len__(self) -> int:
        total = 0
        for i,d in enumerate(self.chaindataset.datasets):
            assert isinstance(
                d, torch.utils.data.IterableDataset
            ), "ChainDataset only supports IterableDataset"

            total += (len(d) + self.batch_size[i] - 1) // self.batch_size[i]
        return total
    
    def __iter__(self) -> Iterator[List[int]]:
        prefix_len = 0
        for i,d in enumerate(self.chaindataset.datasets):
            chunks = (len(d) + self.batch_size[i] - 1) // self.batch_size[i]
            indices = torch.arange(prefix_len, prefix_len + len(d)).chunk(chunks)
            prefix_len += len(d)
            for batch in indices:
                yield batch.tolist()
        





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


def collate_fn_fileD(batch_data, pad_id=2, gist_token_ids=32000):
    # print(len(batch_data), len(batch_data[0]))
    
    input_ids, attention_masks, labels = [], [], []
    for i,batch in batch_data:
        max_length = find_max_length(batch)
        for data in batch:
            # if gist_token_ids in data:
            #     gist_loc = data.index(gist_token_ids)
            # else:
            #     gist_loc = len(data)
            gist_loc = data.index(gist_token_ids)
            difference = max_length - len(data)
            attention_mask = torch.zeros(max_length)
            attention_mask[:len(data)] = 1
            label = [-100] * (gist_loc + 1) + data[(gist_loc + 1):] + [-100] * difference
            data = data + [pad_id] * difference
            input_ids.append(data)
            attention_masks.append(attention_mask.tolist())
            labels.append(label)

    return (i, torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_masks), torch.tensor(labels))


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
