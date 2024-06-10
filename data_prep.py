import argparse
from transformers import AutoTokenizer
import zstandard as zstd
import json
import random
import math
import os
import nltk


def create_ffile():
    with open("slimpajama_train_file.json","w") as f:
        for i in range(0,11):
            prefix = "/apdcephfs_qy3/share_733425/zhisonzhang/users/shared_data/SlimPajama-627B/train/chunk{}".format(i)
            for j in range(5920):
                f_name = "example_train_{}.jsonl.zst".format(j)
                f_path = "{}/{}".format(prefix, f_name)
                if os.path.exists(f_path):
                    f.write(f_path+"\n")
    
    with open("slimpajama_validation_file.json","w") as f:
        for i in range(0,6):
            prefix = "/apdcephfs_qy3/share_733425/zhisonzhang/users/shared_data/SlimPajama-627B/validation/chunk{}".format(i)
            for j in range(6400):
                f_name = "example_holdout_{}.jsonl.zst".format(j)
                f_path = "{}/{}".format(prefix, f_name)
                if os.path.exists(f_path):
                    f.write(f_path+"\n")

    with open("slimpajama_test_file.json","w") as f:
        for i in range(0,6):
            prefix = "/apdcephfs_qy3/share_733425/zhisonzhang/users/shared_data/SlimPajama-627B/test/chunk{}".format(i)
            for j in range(6400):
                f_name = "example_holdout_{}.jsonl.zst".format(j)
                f_path = "{}/{}".format(prefix, f_name)
                if os.path.exists(f_path):
                    f.write(f_path+"\n") 







def main():

    parser = argparse.ArgumentParser(description='Process data.')
    parser.add_argument('--mode', type=str, choices=["ffile", "tokenize"], default="ffile", help='the sample proportion')
    parser.add_argument('--proportion', type=float, default=0.016, help='the sample proportion')
    parser.add_argument('--input_ffile', type=str, help='the name of the input file to process')
    parser.add_argument('--split', type=int, default=200, help='the number of split')
    parser.add_argument('--split_i', type=int, help='the index of split')
    args = parser.parse_args()



    if args.mode == "ffile":
        create_ffile()
    else:
        with open(args.input_ffile, 'r') as f:
            ffiles = f.read().split("\n")

        tokenizer = AutoTokenizer.from_pretrained("/apdcephfs_qy3/share_733425/zhisonzhang/zh/2401mygo/_cache/models--TinyLlama--TinyLlama-1.1B-intermediate-step-1431k-3T/snapshots/036fa4651240b9a1487f709833b9e4b96b4c1574")
        tokenizer.add_special_tokens({"additional_special_tokens": ["<GIST>"]})
        # print(tokenizer.additional_special_tokens, tokenizer.additional_special_tokens_ids)
        tokenizer.pad_token = tokenizer.eos_token
        gist_token_id = tokenizer.additional_special_tokens_ids[-1]
        # gist_loc_token_ids = tokenizer.convert_tokens_to_ids([".","!","?",":",";","..."])
        gist_loc_token_ids = tokenizer.convert_tokens_to_ids(".")
        Slices = [0, 256, 512, 1024]
        
        
        dctx = zstd.ZstdDecompressor()
        interval = math.ceil(len(ffiles)/args.split)
        for input_file in ffiles[args.split_i*interval:(args.split_i+1)*interval]:
            if len(input_file)>0:
                with open(input_file, 'rb') as f:
                    stream = dctx.stream_reader(f)
                    data = stream.read()

                    split_data = data[:].decode('utf-8').split('\n')
                    # print(split_data)
                    
                    Buckets = []
                    for i in range(len(Slices)):
                        Buckets.append([])

                    for item in split_data:
                        if len(item)!=0:
                            item = json.loads(item)
                            sents = nltk.tokenize.sent_tokenize(item['text'])
                            inputs = tokenizer(item['text'] + tokenizer.eos_token)
                            len_s = [0]
                            for i,s in enumerate(sents):
                                s_tokenized = tokenizer(s)
                                len_s.append(len_s[-1]+len(s_tokenized["input_ids"]))
                                if len_s[-1]>128:
                                    break
                                else:
                                    if len_s[-1]>=16:
                                        inputs["input_ids"].insert(len_s[-1]+1, gist_token_id)
                                        break

                            if gist_token_id not in inputs["input_ids"]:
                                inputs["input_ids"].insert(len_s[-1]+1, gist_token_id)


                            # gist_loc = -1
                            # for id in gist_loc_token_ids:
                            #     try:
                            #         gist_loc = inputs["input_ids"].index(id)
                            #         break
                            #     except:
                            #         pass
                            # if gist_loc >= 0:
                            #     inputs["input_ids"].insert(gist_loc+1, gist_token_id)
                            # else:
                            #     inputs["input_ids"].insert(int(len(inputs["input_ids"])*0.1), gist_token_id)

                            for i in range(1, len(Slices)):
                                if len(inputs["input_ids"]) > Slices[i-1] and len(inputs["input_ids"]) <= Slices[i]:
                                    Buckets[i-1].append(inputs["input_ids"])
                            if len(inputs["input_ids"]) > Slices[i]:
                                Buckets[i].append(inputs["input_ids"][:Slices[i]])
                    

                    if "train" in input_file:
                        prefix = "slimpajama_train"
                    elif "validation" in input_file:
                        prefix = "slimpajama_validation"
                    else:
                        prefix = "slimpajama_test"

                    random.seed(12)
                    for i, bucket in enumerate(Buckets):
                        sample_amount = int(args.proportion*len(bucket))
                        print("sample_amount {} for {}".format(sample_amount, Slices[i]))
                        bucket_sample = random.sample(bucket, k=sample_amount)
                        # for input_ids in bucket_sample:
                        #     json_f[i].update({len(json_f[i]): input_ids})
                        for input_ids in bucket_sample:
                            if i<len(Buckets)-1:
                                with open("{}_{}_{}.json".format(prefix, Slices[i], Slices[i+1]), "a") as file:
                                    json.dump({"input_ids":input_ids}, file)
                                    file.write("\n")
                            else:
                                with open("{}_{}_{}.json".format(prefix, Slices[i], "inf"), "a") as file:
                                    json.dump({"input_ids":input_ids}, file)
                                    file.write("\n")



if __name__=="__main__":
    main()
