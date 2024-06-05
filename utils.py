import torch
import math
import json
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
    def __init__(self, lines, start, end, batch_size):
        super(FileIterD).__init__()
        self.lines = lines
        self.start = start
        self.end = end 
        self.batch_size = batch_size
    
    def __len__(self):
        return (self.end - self.start)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        iter_start = self.start
        iter_end = self.end
        if worker_info is not None:
            per_worker = int(math.ceil((self.end-self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(self.start + per_worker, self.end) 

        input_ids = []
        for i,line in enumerate(self.lines):
            if i>=iter_start and i<iter_end:
                py_dict = json.loads(line)
                input_ids.append(py_dict["input_ids"])
        
        for i in range(iter_start, iter_end, self.batch_size):
            yield input_ids[i:i+self.batch_size]
        

        # return iter(input_ids[iter_start:iter_end])



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
    for batch in batch_data:
        max_length = find_max_length(batch)
        for data in batch:
            gist_loc = data.index(gist_token_ids)
            difference = max_length - len(data)
            attention_mask = torch.zeros(max_length)
            attention_mask[:len(data)] = 1
            label = [-100] * (gist_loc + 1) + data[(gist_loc + 1):] + [-100] * difference
            data = data + [pad_id] * difference
            input_ids.append(data)
            attention_masks.append(attention_mask.tolist())
            labels.append(label)

    return (torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(labels))
