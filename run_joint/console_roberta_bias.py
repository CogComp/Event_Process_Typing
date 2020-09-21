import torch
print(torch.cuda.is_available())
from transformers import RobertaTokenizer, RobertaModel, GPT2Model, RobertaForMultipleChoice
import tqdm, sklearn
import numpy as np
import os, time, sys
import pickle
import multiprocessing
from multiprocessing import Process, Value, Manager
from itertools import chain
import scipy, random
import json

if '../utils' not in sys.path:
    sys.path.append('../utils')

from data import Data

from jointSSmrl_roberta_bias import torchpart


import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
def get_hypernym_path(input_word, max_length=20, return_single_set = True):
    paths = list()
    syn_sets = wn.synsets(input_word)
    for syn in syn_sets:
        raw_path = syn.hypernym_paths()
        for p in raw_path:
            tmp_path = [input_word]
            last_node = input_word
            for tmp_synset in p[::-1]:
                tmp_postag = tmp_synset._name.split('.')[1]
                if tmp_postag == 'v':
                    new_node = tmp_synset._name
                else:
                    new_node = tmp_synset._name
                dot_p = new_node.find('.')
                if dot_p > 0:
                    new_node = new_node[:dot_p]
                tmp_path.append(new_node)
                last_node = new_node
            paths.append(tmp_path[:max_length])
    if len(paths) == 0:
        paths = [[input_word]]
    if return_single_set:
        sets = set([])
        for x in paths:
            for y in x:
                sets.add(y)
            return sets
    return paths




def main():
    data_bin, model_bin = '../run/seqVerbMC/data_subsrl_1sv_1sa_argtrim.bin', './full_model/Roberta_BI/full_model_sptoken_ep121_a1.0_m1-0.1_m2-0.1.bin'
    data = Data()
    if os.path.exists(data_bin):
        data.load(data_bin)
        print ("==ATTN== ",len(data.processes)," sequences.")
    else:
        data.load_tsv_plain(data_file)
        data.save(data_bin)
    
    data.dump_dataset_format('./dataset_seq.tsv', 'seq')
    
    # W/O n-1 gram
    M = torchpart()
    M.load(model_bin)
    M.serve_verb([' '], data, limit_ids=None, topk=1), M.serve_arg([' '], data, limit_ids=None, topk=1)
    
    sequence = input("Events split by @ (exit() to quit):")
        #verbs, sequences, true_ids, v2s, limit_ids
    while True:
        sequence = sequence.strip().lower()
        if sequence[:4] == 'exit':
            exit()
        sequence = sequence.split('@')
        vtype, atype = M.serve_verb(sequence, data, limit_ids=None, topk=6), M.serve_arg(sequence, data, limit_ids=None, topk=6)
        print (vtype, atype)
        sequence = input("Events split by @ (exit() to quit):")

if __name__ == "__main__":
    main()


