"""Processing of data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
print(torch.cuda.is_available())
from transformers import BertTokenizer, BertModel, GPT2Model, BertForMultipleChoice, RobertaTokenizer, RobertaModel
import tqdm, sklearn
import numpy as np
import os, time, sys
import pickle
from os import listdir
from os.path import isfile, join
import scipy
import xml.etree.ElementTree as ET


def getsubidx(x, y, begin=0):
    l1, l2 = len(x), len(y)
    for i in range(begin,l1):
        if x[i:i+l2] == y:
            return i
    return None


class WSD_BERT_NN(object):

    def __init__(self):
        # main events
        #self.main_sents = []
        #self.main_events = []
        self.tokenizer = None
        self.model = None
        self.word2synembset = {}
        self.word2pos_syn = {}
        self.word2mfs = None
        self.word2synsentset = {}
        self.sem_cor_sent = []
        self.token_merge_mode = None
        self.pos_map = {'NN':'NOUN', 'JJ':'ADJ', 'VB':'VERB', 'RB':'ADV'}
        
    
    def initialize(self, pretrained='bert-base-uncased', tokenizer='bert-base-uncased', sep_token='[SEP]', annotation_key='lexsn', wn_firstsen_file='../data/ALL.mfs.txt'):
        if pretrained.split('-')[0]=='bert':
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer, sep_token=sep_token)
            self.model = BertModel.from_pretrained(pretrained, output_hidden_states=True)
        elif pretrained.split('-')[0] == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer, sep_token=sep_token)
            self.model = RobertaModel.from_pretrained(pretrained, output_hidden_states=True)
        self.model.cuda()
        assert (annotation_key in ['lexsn', 'wnsn'])
        self.annotation_key = annotation_key
        if wn_firstsen_file is not None:
            self.word2mfs = {}
            for line in open(wn_firstsen_file):
                line = line.strip().split('\t')[-1]
                line = line.split(' ')
                line = line[0].split('%')
                if len(line) == 2:
                    if self.word2mfs.get(line[0]) is None:
                        self.word2mfs[line[0]] = line[1]
    
    def load_and_encode_semcor(self, folder_path='../data/semcor-corpus/semcor/semcor/brownv/tagfiles', token_merge_mode='avg', avg_vec=True, clear=True):
        assert (token_merge_mode in ['avg', 'first'])
        self.token_merge_mode = token_merge_mode
        onlyfiles = [join(folder_path, f) for f in listdir(folder_path) if isfile(join('../data/semcor-corpus/semcor/semcor/brownv/tagfiles', f))]
        if clear:
            self.word2synembset, self.word2synsentset, word2pos_syn, self.sem_cor_sent = {}, {}, {}, []
        sent_id = len(self.sem_cor_sent)
        for f in tqdm.tqdm(onlyfiles):
            tree = ET.parse(f)
            root = tree.getroot()
            for P in root[0]:
                try:
                    sent0 = P[0]
                except:
                    continue
                for S in P:
                    sent, senses, poses = [], [], []
                    tokens = S
                    
                    for T in tokens:
                        this_t = T.attrib.get('lemma')
                        if this_t is None:
                            this_t = T.text.lower()
                        sent.append(this_t)
                        senses.append( T.attrib.get(self.annotation_key) )
                        this_pos = T.attrib.get('pos')
                        if this_pos is not None:
                            this_pos = self.pos_map.get(this_pos)
                        poses.append(this_pos)
                    if len(sent) < 2:
                        continue
                    raw_sent = ' '.join(sent)
                    self.sem_cor_sent.append(raw_sent)
                    tokenized_sent = self.tokenizer.encode(raw_sent, add_special_tokens=False)
                    vecs = self.model(torch.tensor(tokenized_sent).cuda().unsqueeze(0))[0][0].data.cpu().numpy()
                    #print (vecs.shape)
                    begin = 0
                    for i in range(len(sent)):
                        t_sense = senses[i]
                        this_pos = poses[i]
                        if t_sense is not None:
                            #try:
                            # wnsn is a int
                            """
                            if self.annotation_key == 'wnsn':
                                semi_col = t_sense.find(';')
                                if semi_col > 1:
                                    t_sense = t_sense[:semi_col]
                                t_sense = int(t_sense)
                            """
                            t_sense = t_sense.split(';')
                            if self.annotation_key == 'wnsn':
                                t_sense = [int(x) for x in t_sense]
                            #except:
                            #print (f)
                            #exit()
                            t_token = sent[i]
                            tokenized_token = self.tokenizer.encode(t_token, add_special_tokens=False)
                            tid = getsubidx(tokenized_sent, tokenized_token, begin)
                            # if tid==None:
                            #     print(t_token+' is None tid')
                            #     continue
                            assert (tid is not None)
                            begin = tid + len(tokenized_token)
                            if token_merge_mode == 'avg':
                                if len(tokenized_token) > 1:
                                    t_vec = np.average(vecs[tid:tid+len(tokenized_token)], axis=0)
                                else:
                                    t_vec = vecs[tid]
                            else:
                                t_vec = vecs[tid]
                            if self.word2synembset.get(t_token) is None:
                                self.word2synembset[t_token] = {}
                            if self.word2pos_syn.get(t_token) is None:
                                self.word2pos_syn[t_token] = {}
                            # Add sent to self.word2synsentset
                            if self.word2synsentset.get(t_token) is None:
                                self.word2synsentset[t_token] = {}
                            for this_sense in t_sense:
                                if self.word2synembset[t_token].get(this_sense) is None:
                                    self.word2synembset[t_token][this_sense] = [t_vec]
                                else:
                                    self.word2synembset[t_token][this_sense].append(t_vec)
                                if self.word2pos_syn[t_token].get(this_pos) is None:
                                    self.word2pos_syn[t_token][this_pos] = set([this_sense])
                                else:
                                    self.word2pos_syn[t_token][this_pos].add(this_sense)
                                
                                if self.word2synsentset[t_token].get(this_sense) is None:
                                    self.word2synsentset[t_token][this_sense] = [sent_id]
                                else:
                                    self.word2synsentset[t_token][this_sense].append(sent_id)
                    sent_id += 1
                            
        
        if avg_vec:
            for X, senses in self.word2synembset.items():
                for Y, vecs in senses.items():
                    self.word2synembset[X][Y] = [np.average(vecs, axis=0)]
        
        print ("Loaded semcor BERT vectors for", len([x for x,y in self.word2synembset.items()]))
    
    # return number or none
    def get_wn_sense_id(self, token, context, n_occur = 1, token_merge_mode=None):
        assert (context.count(token) >= n_occur)
        if token_merge_mode is None:
            token_merge_mode = self.token_merge_mode
        assert (token_merge_mode in ['first','avg'])
        if self.word2synembset.get(token) is None:
            return None
        tokenized_token = self.tokenizer.encode(token, add_special_tokens=False)
        tokenized_sent = self.tokenizer.encode(context, add_special_tokens=False)
        vecs = self.model(torch.tensor(tokenized_sent).cuda().unsqueeze(0))[0][0].data.cpu().numpy()
        n_seen = 0
        tid = 0
        while tid < len(tokenized_sent):
            if tokenized_sent[tid:tid+len(tokenized_token)] == tokenized_token:
                n_seen += 1
                if n_seen >= n_occur:
                    break
                else:
                    tid += len(tokenized_token)
            else:
                tid += 1
        if token_merge_mode == 'avg':
            if len(tokenized_token) > 1:
                t_vec = np.average(vecs[tid:tid+len(tokenized_token)], axis=0)
            else:
                t_vec = vecs[tid]
        else:
            t_vec = vecs[tid]
        #print (tid, tid+len(tokenized_token))
        m_dist = 2
        sense_id = None
        for sid, Y in self.word2synembset[token].items():
            for v in Y:
                #print (v)
                #print (t_vec)
                #exit()
                n_dist = scipy.spatial.distance.cosine(v, t_vec)
                if n_dist < m_dist:
                    m_dist = n_dist
                    sense_id = sid
        return sense_id
    
    
    def get_wn_sense_id_wpos(self, token, context, n_occur = 1, pos='VB', token_merge_mode=None):
        assert (context.count(token) >= n_occur)
        assert (pos in ['VB', 'NN', 'JJ', 'NNP', 'RB'])
        if token_merge_mode is None:
            token_merge_mode = self.token_merge_mode
        assert (token_merge_mode in ['first','avg'])
        if self.word2synembset.get(token) is None:
            return None
        tokenized_token = self.tokenizer.encode(token, add_special_tokens=False)
        tokenized_sent = self.tokenizer.encode(context, add_special_tokens=False)
        vecs = self.model(torch.tensor(tokenized_sent).cuda().unsqueeze(0))[0][0].data.cpu().numpy()
        n_seen = 0
        tid = 0
        while tid < len(tokenized_sent):
            if tokenized_sent[tid:tid+len(tokenized_token)] == tokenized_token:
                n_seen += 1
                if n_seen >= n_occur:
                    break
                else:
                    tid += len(tokenized_token)
            else:
                tid += 1
        if token_merge_mode == 'avg':
            if len(tokenized_token) > 1:
                t_vec = np.average(vecs[tid:tid+len(tokenized_token)], axis=0)
            else:
                t_vec = vecs[tid]
        else:
            t_vec = vecs[tid]
        #print (tid, tid+len(tokenized_token))
        m_dist = 2
        sense_id = None
        pos_sense_set = self.word2pos_syn.get(token)
        if pos_sense_set is not None:
            pos_sense_set = pos_sense_set.get(pos)
        if pos_sense_set is None:
            # no annotation for the same POS, then use all senses
            pass
        for sid, Y in self.word2synembset[token].items():
            if pos_sense_set is None or sid in pos_sense_set:
                for v in Y:
                    #print (v)
                    #print (t_vec)
                    #exit()
                    n_dist = scipy.spatial.distance.cosine(v, t_vec)
                    if n_dist < m_dist:
                        m_dist = n_dist
                        sense_id = sid
        return sense_id
    
    
    
    def get_wn_first_sen(self, token):
        return self.word2mfs.get(token)
    
    def get_semcor_sense_contexts(self, word, sense_id):
        rst = []
        sentset = self.word2synsentset.get(word)
        if sentset is not None:
            sentset = sentset.get(sense_id)
            if sentset is not None:
                for i in sentset:
                    rst.append(self.sem_cor_sent[i])
        return rst
        

    def save(self, filename):
        f = open(filename,'wb')
        pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        print("Save WSD-BERT-NN object as", filename)
    def load(self, filename):
        f = open(filename,'rb')
        tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
        print("Loaded WSD-BERT-NN object from", filename)


def main():
    wsd_file = 'wsd_bert_nn.bin'
    wsd = WSD_BERT_NN()
    if os.path.exists(wsd_file):
        wsd.load(wsd_file)
        print ("==ATTN== ", len([x for x,y in wsd.word2synembset.items()]))
    else:
        wsd.initialize(annotation_key='lexsn')
        wsd.load_and_encode_semcor(folder_path='../data/semcor-corpus/semcor/semcor/brownv/tagfiles', token_merge_mode='avg', avg_vec=True)
        wsd.save(wsd_file)
    
    print ("..Running test for WSD-BERT-NN..")
    print ("the rain *bank* the soil up behind the gate  ", wsd.get_wn_sense_id('bank',"the rain bank the soil up behind the gate", 1))
    print ("I pay the money straight into my *bank*  ", wsd.get_wn_sense_id('bank', "I pay the money straight into my bank", 1))
    print ("and I run, I *run* so far away  ", wsd.get_wn_sense_id('run', "and I run, I run so far away", 2))
    print ("the second *run* of the practice still failed  ", wsd.get_wn_sense_id('run', "the second run of the practice still failed", 1))
    
    print ("the rain *bank* the soil up behind the gate  ", wsd.get_wn_sense_id_wpos('bank',"the rain bank the soil up behind the gate", 1, 'VB'))
    print ("I pay the money straight into my *bank*  ", wsd.get_wn_sense_id_wpos('bank', "I pay the money straight into my bank", 1, 'NN'))
    print ("and I run, I *run* so far away  ", wsd.get_wn_sense_id_wpos('run', "and I run, I run so far away", 2, 'VB'))
    print ("the second *run* of the practice still failed  ", wsd.get_wn_sense_id_wpos('run', "the second run of the practice still failed", 1, 'NN'))
    
    
if __name__ == "__main__":
    main()