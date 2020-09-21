import torch
print(torch.cuda.is_available())
from transformers import BertTokenizer, BertModel, GPT2Model, BertForMultipleChoice
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
from wsd import WSD_BERT_NN

"""
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)

last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
"""

"""
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMultipleChoice.from_pretrained('bert-base-uncased')
choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]

input_ids = torch.tensor([tokenizer.encode(s, add_special_tokens=True) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
outputs = model(input_ids, labels=labels)
loss, classification_scores = outputs[:2]
"""

class torchpart(object):

    def __init__(self):
        self.model=None
        self.tokenizer=None
        self.batch_size=128
        self.epsilon = 1e-8
        self.epoch=1
    
    def initialize(self, pretrained='bert-base-uncased', tokenizer='bert-base-uncased', sep_token='[SEP]'):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer, sep_token=sep_token)
        self.model = BertModel.from_pretrained(pretrained, output_hidden_states=True)
        self.model.cuda()
        #self._M2 = torch.nn.Linear(768*2, 768).cuda()
        self._M = torch.nn.Linear(768*2, 1).cuda()
        self._loss = torch.nn.BCELoss().cuda()
        self.bos_token = '[CLS] '
        self.sep_token = ' ' + sep_token + ' '
    
    # sequences are already with [SEP], and partitioned into training sets.
    def train_verb(self, verbs, sequences, true_senses, all_senses, epochs=5, learning_rate=0.001):
        #all_cases = [[self.sep_token.join([s, v]) for v in verbs] for s in sequences]
        all_cases = [s for s in sequences]
        true_senses = [x for x in true_senses]
        all_senses = [x for x in all_senses]
        #print (true_senses[:3])
        #print (all_cases[0])
        assert (len(all_cases) == len(true_senses))
        print ("Begin training with ", len(all_cases), " cases and ", len(verbs), " choices.")
        
        #params = [x for x in self.model.parameters()] + [self._M]
        
        optimizer = torch.optim.Adam(chain(self.model.parameters(), self._M.parameters()), lr=learning_rate, amsgrad=True)#torch.optim.Adam(chain(self.model.parameters(), self._M.parameters()), lr=learning_rate)
        
        indicator = torch.tensor(np.ones(self.batch_size, dtype=np.float32), requires_grad=False).cuda()
        n_indicator = torch.tensor(np.zeros(self.batch_size, dtype=np.float32), requires_grad=False).cuda()
        
        for epoch in range(epochs):
            print ("Begin epoch #", self.epoch)
            l = len(all_cases)
            indices = np.arange(l)
            np.random.shuffle(indices)
            this_cases, this_verbs = [all_cases[i] for i in np.concatenate([indices, indices[0:self.batch_size]])], [true_senses[i] for i in np.concatenate([indices, indices[0:self.batch_size]])]
            #step = random.randint(len(all_senses))
            false_verbs = []
            for i in tqdm.tqdm(range(len(this_verbs))):
                this_step = random.randint(0, len(all_senses))
                while len(all_senses[(this_step) % len(all_senses)]) == len(this_verbs[i]) and all_senses[(this_step) % len(all_senses)][:9] == this_verbs[i][:9]:
                    this_step += 1
                false_verbs.append(all_senses[(this_step) % len(all_senses)])
            
            this_loss = []
            for b in tqdm.tqdm(range(0, l, self.batch_size)):
                batch_cases, batch_verbs, batch_false = [this_cases[i] for i in range(b, b + self.batch_size)], [this_verbs[i] for i in range(b, b + self.batch_size)], [false_verbs[i] for i in range(b, b + self.batch_size)]
                input_ids = torch.tensor([self.tokenizer.encode(ss, add_special_tokens=True, max_length=50, pad_to_max_length=True) for ss in batch_cases]).cuda()

                input_verbs = torch.tensor([self.tokenizer.encode(ss, add_special_tokens=True, max_length=50, pad_to_max_length=True) for ss in batch_verbs]).cuda()
                
                input_false = torch.tensor([self.tokenizer.encode(ss, add_special_tokens=True, max_length=50, pad_to_max_length=True) for ss in batch_false]).cuda()

                outputs = torch.mean(self.model(input_ids)[0], 1)
                output_verbs = torch.mean(self.model(input_verbs)[0], 1)
                output_false = torch.mean(self.model(input_false)[0], 1)
                #print ([x.shape for x in self.model(input_ids)])
                #print (outputs.shape, output_verbs.shape)
                
                #loss1 = self._M(torch.sub(outputs, output_verbs).abs()).squeeze()
                loss1 = torch.sigmoid(self._M(torch.cat((torch.sub(outputs, output_verbs).abs(), torch.mul(outputs, output_verbs)), 1)).squeeze())
                loss1 = self._loss(loss1, indicator)
                
                loss2 = torch.sigmoid(self._M(torch.cat((torch.sub(outputs, output_false).abs(), torch.mul(outputs, output_false)), 1)).squeeze())
                loss2 = self._loss(loss2, n_indicator)
                
                loss = torch.add(loss1, loss2)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                this_loss.append(loss.data.cpu().numpy())
            
            this_loss = np.average(this_loss)
            print ("Loss = ", this_loss)
            self.epoch += 1
            if np.isnan(this_loss):
                exit()
        
                
    def test_verb(self, verbs, sequences, true_ids, v2s, limit_ids=True):
        cand_ids = set([])
        if limit_ids:
            for id in true_ids:
                cand_ids.add(id)
        else:
            cand_ids = set([i for i in range(len(verbs)) if v2s.get(verbs[i]) is not None])
        all_cases = [s for s in sequences]
        senses = [v2s[v] if v2s.get(v) is not None else [' '] for v in verbs]
        #true_verbs = [verbs[x] for x in true_ids]
        assert (len(all_cases) == len(true_ids))
        print ("Begin testing with ", len(all_cases), " case.")
        
        cpu_count = multiprocessing.cpu_count()
        manager = Manager()
        mrr, hits1, hits10 = manager.list(), manager.list(), manager.list()
        
        index = Value('i',0,lock=True)
        
        #self._M.cpu()
        #self.model.cpu()
        
        s_vec = np.array([torch.mean(self.model(torch.tensor(self.tokenizer.encode(ss, add_special_tokens=True, max_length=50, pad_to_max_length=True)).cuda().unsqueeze(0))[0], -2).data.cpu().numpy()[0] for ss in tqdm.tqdm(all_cases)])
        v_vec = [[torch.mean(self.model(torch.tensor(self.tokenizer.encode(ss, add_special_tokens=True, max_length=50, pad_to_max_length=True)).cuda().unsqueeze(0))[0], -2).data.cpu().numpy()[0] for ss in sset] for sset in tqdm.tqdm(senses)]
        print (s_vec.shape)
        
        self._M.cpu()
        #self._M2.cpu()
        #W1, b1, W2, b2 = self._M.weight.data.numpy(), self._M.bias.data.numpy(), self._M2.weight.data.numpy(), self._M2.bias.data.numpy()
        W1, b1 = self._M.weight.data.numpy(), self._M.bias.data.numpy()   

        t0 = time.time()
        def test(s_vec, v_vec, cand_ids):
            while index.value < len(all_cases):
                id = index.value
                index.value += 1
                if id < 10 or id % int(len(all_cases) / 25) == 0:
                    print ('At ',id)
                if id >= len(all_cases):
                    return
                #if id % 1000 == 0:
                    #print (id ,'/', len(all_cases), ' time used ',time.time() - t0)
                tid = true_ids[id]
                this_s, this_v = s_vec[id], v_vec[tid]
                t_dist = float('-inf')
                for cur_v in this_v:
                    c_dist = np.dot(W1, np.concatenate([this_s, cur_v])) + b1
                    if t_dist < c_dist:
                        t_dist = c_dist
                rank = 1
                for i in cand_ids:
                    s_dist = float('-inf')
                    if i != tid:
                        for cur_v in v_vec[i]:
                            c_dist = np.dot(W1, np.concatenate([this_s, cur_v])) + b1
                            if s_dist < c_dist:
                                s_dist = c_dist
                        if t_dist < s_dist:
                            rank +=1
                h1, h10 = 0., 0.
                if rank < 11:
                    h10 = 1.
                    if rank < 2:
                        h1 = 1.
                mrr.append(1. / rank)
                hits1.append(h1)
                hits10.append(h10)
        
        processes = [Process(target=test, args=(s_vec, v_vec, cand_ids)) for x in range(8)]
        
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        
        mrr, hits1, hits10 = np.average(mrr), np.average(hits1), np.average(hits10)
        print (mrr, hits1, hits10)
        self._M.cuda()
        return mrr, hits1, hits10
        
    def save(self, filename):
        self.predictor = None
        f = open(filename,'wb')
        pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        print("Save data object as", filename)
    
    def load(self, filename):
        f = open(filename,'rb')
        tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
        print("Loaded data object from", filename)


def main():
    data_file, data_bin, model_bin, rst_file = '../data/wikihow_process/wikiHowSubsequences.tsv', '../data/wikihow_process/data_subsrl_1sv_1sa_argtrim.bin', './seqSSMLP_subsrl_wsd/BertVerbMC/', './seqSSMLP_subsrl_wsd/results_subsrl_wsd_' + time.strftime('%H_%d-%m-%y') + '.txt'
    skip_training = False
    if len(sys.argv) > 1:
        skip_training = bool(sys.argv[1])
    try:
        os.mkdir('./seqSSMLP_subsrl_wsd')
        os.mkdir('./seqSSMLP_subsrl_wsd/BertVerbMC/')
    except:
        pass
    data = Data()
    if os.path.exists(data_bin):
        data.load(data_bin)
        print ("==ATTN== ",len(data.processes)," sequences.")
    else:
        data.load_tsv_plain(data_file)
        data.save(data_bin)
    
    wsd_file = '../utils/wsd_bert_nn.bin'
    wsd = WSD_BERT_NN()
    if os.path.exists(wsd_file):
        wsd.load(wsd_file)
        print ("==ATTN== ", len([x for x,y in wsd.word2synembset.items()]))
    else:
        wsd.initialize(annotation_key='wnsn')
        wsd.load_and_encode_semcor(folder_path='../data/semcor-corpus/semcor/semcor/brownv/tagfiles', token_merge_mode='avg')
        wsd.save(wsd_file)
    
    # W/O n-1 gram
    sequences = data.join_batch_sent(data.processes)
    r_verbs, r_args = {y: x for x, y in data.verb_vocab.items()}, {y: x for x, y in data.arg_vocab.items()}
    n_verbs, n_args = len([x for x, y in data.verb_vocab.items()]), len([x for x, y in data.arg_vocab.items()])
    #print (n_verbs)
    verbs, args = [r_verbs[x] for x in range(n_verbs)], [r_args[x] for x in range(n_args)]
    vid, aid = np.array(data.verb_id), np.array(data.arg_id)
    assert (len(vid) == len(aid))
    verb_list, arg_list = [verbs[x] for x in vid], [args[x] for x in aid]
    wnsn_list = []
    print ("Performing WSD.")
    for i in tqdm.tqdm(range(len(vid))):
        sid = wsd.get_wn_sense_id(verb_list[i],verb_list[i] + ' ' + arg_list[i], 1)
        if sid is None:
            sid = 1 # First sense by default
        wnsn_list.append(sid)
    true_senses = [data.v2s[verbs[x]] for x in vid]
    flatten_senses = []
    for x in true_senses:
        for y in x:
            flatten_senses.append(y)
    all_senses = tuple(set(flatten_senses))
    for i in range(len(true_senses)):
        if len(true_senses[i]) > wnsn_list[i]:
            true_senses[i] = true_senses[i][wnsn_list[i]]
        else:
            true_senses[i] = true_senses[i][0]
    #print (true_senses[:3])
    indices = np.arange(len(sequences))
    
    max_fold = 1
    rs = sklearn.model_selection.ShuffleSplit(n_splits=max_fold, test_size=0.1, random_state=777)
    
    avg_mrr, avg_hits1, avg_hits10 = [], [], []
    print (len(verbs))
    
    with open(rst_file, 'w') as fp:
        fp.write('Total processes ' + str(len(sequences)) + '; verbs ' + str(len(verbs)) + '  ; args ' + str(len([x for x, y in data.arg_vocab.items()])) + '\n\n')
        fold = 1
        for train_index, test_index in rs.split(indices):
            train_seq, test_seq = [sequences[x] for x in train_index], [sequences[x] for x in test_index]
            train_senses, test_vid = [true_senses[x] for x in train_index], vid[test_index]
            M = torchpart()
            M.initialize()
            if not skip_training:
                for ep in range(10):
                #verbs, sequences, senses, epochs
                    M.train_verb(verbs, train_seq, train_senses, all_senses, epochs=4, learning_rate=0.0001)
                    M.save('./seqSSMLP_subsrl_wsd/BertVerbMC/tmp_fold'+str(fold)+'_'+str(ep*5+5)+'.bin')
                    mrr, hits1, hits10 = M.test_verb(verbs, test_seq, test_vid, data.v2s, limit_ids=True)
                    avg_mrr.append(mrr)
                    avg_hits1.append(hits1)
                    avg_hits10.append(hits10)
                    fp.write("Fold "+str(fold) + 'Epochs ' + str(ep*5+5) + '  mrr=' + str(mrr) + '  hits@1=' + str(hits1) + '  hits@10=' + str(hits10) + '\n')
            else:
                M.load('./seqSSMLP_subsrl_wsd/BertVerbMC/tmp_fold'+str(fold)+'.bin')
            #verbs, sequences, true_ids, v2s, limit_ids
            del M
            fold += 1
        fp.write('Avg: mrr='+str(np.average(avg_mrr)) + '  hits@1=' + str(np.average(avg_hits1)) + '  hits@10=' + str(np.average(avg_hits10)) + '\n')

if __name__ == "__main__":
    main()


