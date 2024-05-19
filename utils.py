import torch
import math
import numpy as np

# class MyIterableDataset(torch.utils.data.IterableDataset):
#     def __init__(self, start, end):
#         super(MyIterableDataset).__init__()
#         assert end > start, "this example code only works with end >= start"
#         self.start = start
#         self.end = end
#     def __iter__(self):
#         worker_info = torch.utils.data.get_worker_info()
#         if worker_info is None:  # single-process data loading, return the full iterator
#             iter_start = self.start
#             iter_end = self.end
#         else:  # in a worker process
#             # split workload
#             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
#             worker_id = worker_info.id
#             iter_start = self.start + worker_id * per_worker
#             iter_end = min(iter_start + per_worker, self.end)
#         return iter(range(iter_start, iter_end))
    



class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, mem_arr, gist_activations, gist_location_id, config):
        super(MyIterableDataset).__init__()
        self.arr = mem_arr
        self.input_ids = chunk_arr(mem_arr)
        self.start = 0
        self.end = len(self.input_ids)
        self.gist_activations = gist_activations
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
                input_id.insert(gist_loc, 32000)
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
        return self.end
    



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


