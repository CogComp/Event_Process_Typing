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
    v_thres, l_thres = 50, 5
    if len(sys.argv) > 1:
        v_thres, l_thres = int(sys.argv[1]), int(sys.argv[2])
    #data_file, data_bin, model_bin, test_file = '/shared/corpora-tmp/wikihow/wikiHowSubsequences.tsv', '../run/seqVerbMC/data_subsrl_1sv_1sa_argtrim.bin', './seqSSmrl_subsrl/RobertaVerbMC/tmp_fold_ep151_a1.0_m1-0.1_m2-0.1.bin', './seqSSmrl_subsrl/RobertaVerbMC/test_fold_ep151_a1.0_m1-0.1_m2-0.1.txt'
    data_file, data_bin, model_bin, test_file = '/shared/corpora-tmp/wikihow/wikiHowSubsequences.tsv', '../run/seqVerbMC/data_subsrl_1sv_1sa_argtrim.bin', './seqSSmrl_subsrl/RobertaVerbMC/tmp_fold_ep151_a1.0_m1-0.1_m2-0.1.bin', '../process/recover_test_index_fold1.txt'
    data = Data()
    if os.path.exists(data_bin):
        data.load(data_bin)
        print ("==ATTN== ",len(data.processes)," sequences.")
    else:
        data.load_tsv_plain(data_file)
        data.save(data_bin)
    
    # W/O n-1 gram
    sequences = data.join_batch_sent(data.processes, begin='<s> ', sep=' </s> ')
    seq_len = np.array([len(x) for x in data.processes])
    r_verbs = {y: x for x, y in data.verb_vocab.items()}
    n_verbs = len([x for x, y in data.verb_vocab.items()])
    #print (n_verbs)
    verbs = [r_verbs[x] for x in range(n_verbs)]
    vid = np.array(data.verb_id)
    true_senses = [data.v2s[verbs[x]] for x in vid]
    
    r_args = {y: x for x, y in data.arg_vocab.items()}
    n_args = len([x for x, y in data.arg_vocab.items()])
    #print (n_args)
    args = [r_args[x] for x in range(n_args)]
    aid = np.array(data.arg_id)
    true_arg_senses = [data.a2s[args[x]] for x in aid]
    
    #print (true_senses[:3])
    max_fold = 1
    rs = sklearn.model_selection.ShuffleSplit(n_splits=max_fold, test_size=0.1, random_state=777)
    
    avg_mrr, avg_hits1, avg_hits10 = [], [], []
    avg_mrra, avg_hits1a, avg_hits10a = [], [], []
    print (len(verbs), len(args))
    
    test_index = []
    for x in open(test_file):
        test_index.append(int(x.strip()))
    test_index = np.array(test_index)
    
    
    test_seq = [sequences[x] for x in test_index]
    test_vid = vid[test_index]
    test_aid = aid[test_index]
    M = torchpart()
    M.load(model_bin)
        #verbs, sequences, true_ids, v2s, limit_ids
    M.profile_test_verb(verbs, test_seq, seq_len[test_index], test_vid, data.v2s, v_thres, l_thres)
    M.profile_test_verb(verbs, test_seq, seq_len[test_index], test_vid, data.v2s, 525, 2)
    M.profile_test_verb(verbs, test_seq, seq_len[test_index], test_vid, data.v2s, 425, 2)
    #M.profile_test_verb(verbs, test_seq, seq_len[test_index], test_vid, data.v2s, 130, 3)
    #M.profile_test_verb(verbs, test_seq, seq_len[test_index], test_vid, data.v2s, 100, 4)
 

if __name__ == "__main__":
    main()


