"""Processing of data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pickle, tqdm
import time, sys
import csv
from sklearn.svm import SVR
from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from absl import flags

FLAGS = flags.FLAGS
FLAGS(sys.argv)

flags.DEFINE_string('srl_predictor', 'https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz', 'SRL predictor on AWS.')

class Data(object):
    '''The abstract class that defines interfaces for holding all data.
    '''

    def __init__(self):
        # main events
        #self.main_sents = []
        #self.main_events = []
        self.main_id = {}
        self.true_id = []
        self.verb_vocab = {}
        self.verb_id = []
        self.arg_vocab = {}
        self.arg_id = []
        self.id2verb, self.id2arg = {}, {}
        #self.alternatives = []
        # processes
        self.processes = []
        # srl
        self.predictor = None
        self.v2s = {}
        self.a2s = {}
        # tokenizer

    def load_tsv_plain(self, filepath='./wikiHowSubsequences.tsv'):
        skip_first=True
        id, vid, aid = 0, 0, 0
        num_main = 0
        cur_subproc = {}
        if self.predictor is None:
            self.predictor = Predictor.from_path(FLAGS.srl_predictor)
        prev_raw = None
        processed = 0
        inform_open = tqdm.tqdm(open(filepath, encoding='utf8'))
        for line in inform_open:
            processed += 1
            #if processed % 100 == 0:
                #print ('processed ',processed, " stored ", len(self.processes))
            inform_open.set_postfix(preserved=len(self.processes))
            if skip_first:
                skip_first = False
                continue
            line = line.strip().lower().split('\t')
            if len(line) < 3:
                continue
            if line[0].find('how to ') == 0:
                this_raw = line[0]
                line[0] = line[0][7:]
                #if processed < 10:
                    #print (line[0])
                try:
                    srl = self.predictor.predict(line[0])
                    #if processed < 10:
                        #print (srl)
                    #srl=srl[0]
                except:
                    continue
                verb, arg1 = [], []
                
                if srl.get('verbs') is None or len(srl.get('verbs')) == 0:
                    continue
                for i in range(len(srl['verbs'][0]['tags'])):
                    if srl['verbs'][0]['tags'][i] == 'B-V':
                        verb.append(srl['words'][i])
                    elif srl['verbs'][0]['tags'][i] in ['B-ARG1', 'I-ARG1']:
                        arg1.append(srl['words'][i])

                if len(verb) != 1 or len(arg1) == 0 or len(arg1) > 4:
                    continue
                verb, arg1 = verb[0], ' '.join(arg1)
                line[0] = verb + ' ' + arg1
            else:
                continue
            this_id = self.main_id.get(line[0])
            if this_id is None:
                self.main_id[line[0]] = id
                self.true_id.append(id)
                id += 1
                num_main += 1
                cur_subproc[line[0]] = line[1]
                #self.alternatives.append(line[1])
                #self.main_sents.append(line[0])
                #self.main_events.append((verb, arg1))
                self.processes.append([line[2]])
                # store verb and arg ids
                this_vid, this_aid = self.verb_vocab.get(verb), self.arg_vocab.get(arg1)
                if this_vid is None:
                    self.verb_id.append(vid)
                    self.verb_vocab[verb] = this_vid = vid
                    vid +=1
                else:
                    self.verb_id.append(this_vid)
                self.id2verb[this_vid] = verb
                if this_aid is None:
                    self.arg_id.append(aid)
                    self.arg_vocab[arg1] = this_aid = aid
                    aid +=1 
                else:
                    self.arg_id.append(this_aid)
                self.id2arg[this_aid] = arg1
            # New alternative.
            elif cur_subproc[line[0]] != line[1] or prev_raw != this_raw:
                if prev_raw != this_raw:
                    prev_raw = this_raw
                #self.main_id[line[0]].append( id )
                #id += 1
                self.true_id.append(this_id)
                cur_subproc[line[0]] = line[1]
                #self.alternatives.append(line[1])
                #self.main_sents.append(line[0])
                #self.main_events.append((verb, arg1))
                self.processes.append([line[2]])
                # store verb and arg ids
                this_vid, this_aid = self.verb_vocab.get(verb), self.arg_vocab.get(arg1)
                if this_vid is None:
                    self.verb_id.append(vid)
                    self.verb_vocab[verb] = this_vid = vid
                    vid +=1
                else:
                    self.verb_id.append(this_vid)
                self.id2verb[this_vid] = verb
                if this_aid is None:
                    self.arg_id.append(aid)
                    self.arg_vocab[arg1] = this_aid = aid
                    aid +=1 
                else:
                    self.arg_id.append(this_aid)
                self.id2arg[aid] = arg1
            else:
                self.processes[-1].append(line[2])
        print ("\n=== Loaded " + str(len(self.processes)) + ' processes for ' + str(num_main) + " events. ===\n")
    
    
    
    def parse_subevent(self):
        processes, true_id, verb_id, arg_id = [], [], [], []
        self.process_vid, self.process_aid = [], []
        # self.verb_vocab = {}, self.arg_vocab = {}, self.id2verb, self.id2arg = {}, {}
        vid, aid = len([x for x, y in self.verb_vocab.items()]), len([x for x, y in self.arg_vocab.items()])
        inform_open = tqdm.tqdm(range(len(self.processes)))
        if self.predictor is None:
            self.predictor = Predictor.from_path(FLAGS.srl_predictor)
        for id in inform_open:
            new_v, new_a, new_line = [], [], []
            inform_open.set_postfix(preserved=len(processes))
            breaked = False
            for event in self.processes[id]:
                if event[-1] in ['.', '!']:
                    event = event[:-1]
                try:
                    srl = self.predictor.predict(event)
                    #if processed < 10:
                        #print (srl)
                    #srl=srl[0]
                except:
                    breaked = True
                    if id < 5:
                        print ('==ATTN==',event)
                    break
                verb, arg1 = [], []
                
                if srl.get('verbs') is None or len(srl.get('verbs')) == 0:
                    breaked = True
                    break
                for i in range(len(srl['verbs'][0]['tags'])):
                    if srl['verbs'][0]['tags'][i] == 'B-V':
                        verb.append(srl['words'][i])
                    elif srl['verbs'][0]['tags'][i] in ['B-ARG1', 'I-ARG1']:
                        arg1.append(srl['words'][i])

                if len(verb) != 1 or len(arg1) == 0 or len(arg1) > 4:
                    breaked = True
                    break
                verb, arg1 = verb[0], ' '.join(arg1)
                line = verb + ' ' + arg1
                new_v.append(verb)
                new_a.append(arg1)
                new_line.append(line)
            """
            if id < 5:
                print (self.processes[id])
                #print ([(x, y) for x, y in self.id2arg.items()])
                print (self.id2verb[self.verb_id[id]], self.id2arg[self.arg_id[id]], new_line, new_v, new_a)
            """
            if not breaked:
                this_vid, this_aid = [], []
                for x in new_v:
                    i = self.verb_vocab.get(x)
                    if i is None:
                        i = vid
                        vid += 1
                        self.verb_vocab[x] = i
                        self.id2verb[i] = x
                    this_vid.append(i)
                for x in new_a:
                    i = self.arg_vocab.get(x)
                    if i is None:
                        i = aid
                        aid += 1
                        self.arg_vocab[x] = i
                        self.id2arg[i] = x
                    this_aid.append(i)
                processes.append(new_line)
                true_id.append(self.true_id[id])
                verb_id.append(self.verb_id[id])
                arg_id.append(self.arg_id[id])
                self.process_vid.append(this_vid)
                self.process_aid.append(this_aid)
            
        self.processes, self.true_id, self.verb_id, self.arg_id = processes, true_id, verb_id, arg_id
        print ("\n=== Parsed " + str(len(self.processes)) + " processes. ===\n")
    
    
    def arg_copy_rate(self):
        hit = 0.
        assert (len(self.arg_id) == len(self.process_aid))
        for i in range(len(self.arg_id)):
            aid = self.arg_id[i]
            if aid in self.process_aid[i]:
                hit += 1.
        return (hit / len(self.arg_id))
    
    def verb_copy_rate(self):
        hit = 0.
        assert (len(self.verb_id) == len(self.process_vid))
        for i in range(len(self.verb_id)):
            vid = self.verb_id[i]
            if vid in self.process_vid[i]:
                hit += 1.
        return (hit / len(self.verb_id))


    
    def trimming_args(self):
        import string
        stop = stopwords.words('english')
        wnl = WordNetLemmatizer()
        dpc = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")
        new_arg_vocab, bridge = {}, {}
        id = 0
        
        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False
        
        def is_puct(s):
            for x in s:
                if x not in string.punctuation:
                    return False
            return True
        
        for x, y in tqdm.tqdm(self.arg_vocab.items()):
            px = dpc.predict(sentence=x)
            rt = px['hierplane_tree']['root']
            wd = rt['word']
            att = rt.get('attributes')
            if att is not None and 'NOUN' in att:
                wd = wnl.lemmatize(wd)
            urls = ['.com', 'www.', 'http', '.amazon', '.tw', '.org', '.cn']
            files = ['.sys', '.iso', '.exe', '.bin', 'jpg']
            for c in urls:
                if wd.find(c) > -1:
                    wd = 'url'
                    break
            for c in files:
                if wd.find(c) > -1:
                    wd = 'file'
                    break
            if wd.find('.') > 0:
                wd = wd.split('.')
                if wd[0] in stop and len(wd) > 1:
                    wd = wd[1]
                else:
                    wd = wd[0]
            elif wd.find('('):
                wd = wd.split('(')
                if wd[0] in stop and len(wd) > 1:
                    wd = wd[1]
                else:
                    wd = wd[0]
            elif wd.find('/'):
                wd = wd.split('/')
                if wd[0] in stop and len(wd) > 1:
                    wd = wd[1]
                else:
                    wd = wd[0]
            if is_number(wd):
                wd = 'number'
            elif is_puct(wd):
                wd = 'punctuation'
            elif wd in stop or len(wd) <= 2:
                wd = 'word'
            this_id = new_arg_vocab.get(wd)
            if this_id is None:
                this_id = id
                id += 1
                new_arg_vocab[wd] = this_id
            bridge[y] = this_id
        new_id2arg = {y:x for x,y in new_arg_vocab.items()}
        new_arg_id, new_process_aid = [], []
        for x in self.arg_id:
            new_arg_id.append(bridge[x])
        for x in self.process_aid:
            new_process_aid.append([bridge[y] for y in x])
        self.arg_vocab, self.id2arg, self.arg_id, self.process_aid = new_arg_vocab, new_id2arg, new_arg_id, new_process_aid
        print("=== Arg trimmed to #", len([x for x,y in self.arg_vocab.items()]))
    
    
    def align_verb_senses_mix(self, filepath='../data/wordnet/sense_verb.tsv', mode='mix', limit=2):
        assert (mode in ['mix', 'first', 'list', 'limit'])
        w2s = {}
        for line in open(filepath):
            line = line.strip().split('\t')
            if len(line) < 2:
                continue
            if w2s.get(line[0]) is None:
                if mode == 'list':
                    w2s[line[0]] = [line[1]]
                else:
                    w2s[line[0]] = line[1]
            else:
                if mode == 'list':
                    w2s[line[0]].append(line[1])
                elif mode == 'mix':
                    w2s[line[0]] += ' [SEP] ' + line[1]
                elif mode == 'first':
                    continue
                elif mode == 'limit' and w2s[line[0]].count(' [SEP] ') + 1 < limit:
                    w2s[line[0]] += ' [SEP] ' + line[1]
        self.v2s_mode = mode
        
        
        processes, true_id, verb_id, arg_id, process_vid, process_aid, pw2s = [], [], [], [], [], [], {}
        inform_open = tqdm.tqdm(range(len(self.verb_id)))
        missed = 0
        for id in inform_open:
            inform_open.set_postfix(missed=missed)
            verb = self.id2verb[self.verb_id[id]]
            sense = w2s.get(verb)
            if sense is None:
                missed += 1
                continue
            if pw2s.get(verb) is None:
                pw2s[verb] = sense
            processes.append(self.processes[id])
            true_id.append(self.true_id[id])
            verb_id.append(self.verb_id[id])
            arg_id.append(self.arg_id[id])
            process_vid.append(self.process_vid[id])
            process_aid.append(self.process_aid[id])
        self.v2s = pw2s
        self.processes, self.true_id, self.verb_id, self.arg_id, self.process_vid, self.process_aid = processes, true_id, verb_id, arg_id, process_vid, process_aid
        print ("Preserved ", len(self.processes), " processes & ", len([x for x, y in self.v2s.items()]), " senses.")
    
    
    
    
    def align_arg_senses_mix(self, filepath='../data/wordnet/sense_wn.tsv', mode='mix', limit=2):
        assert (mode in ['mix', 'first', 'list', 'limit'])
        w2s = {}
        for line in open(filepath):
            line = line.strip().split('\t')
            if len(line) < 2:
                continue
            if w2s.get(line[0]) is None:
                if mode == 'list':
                    w2s[line[0]] = [line[1]]
                else:
                    w2s[line[0]] = line[1]
            else:
                if mode == 'list':
                    w2s[line[0]].append(line[1])
                elif mode == 'mix':
                    w2s[line[0]] += ' [SEP] ' + line[1]
                elif mode == 'first':
                    continue
                elif mode == 'limit' and w2s[line[0]].count(' [SEP] ') + 1 < limit:
                    w2s[line[0]] += ' [SEP] ' + line[1]
        self.a2s_mode = mode
        
        arg_num = len([x for x, y in self.arg_vocab.items()])
        if self.arg_vocab.get('default') is None:
            self.arg_vocab['default'] = arg_num
            self.id2arg[arg_num] = 'default'
            arg_num += 1
        pw2s = {}
        inform_open = tqdm.tqdm(range(len(self.arg_id)))
        missed = 0
        for id in inform_open:
            inform_open.set_postfix(missed=missed)
            arg = self.id2arg[self.arg_id[id]]
            sense = w2s.get(arg)
            if sense is None:
                missed += 1
                #continue
                sense = w2s['default']
            if pw2s.get(arg) is None:
                pw2s[arg] = sense
        self.a2s = pw2s
        print ("Preserved ", len([x for x, y in self.a2s.items()]), " arg senses.")
    
    def dump_verb_arg_distribution(self, ofile1="./verb_distribution.txt", ofile2="./arg_distribution.txt"):
        vd, ad = {}, {}
        vcount, acount = 0, 0
        for v in self.verb_id:
            if vd.get(v) is None:
                vcount += 1
                vd[v] = 1
            else:
                vd[v] += 1
        for a in self.arg_id:
            if ad.get(a) is None:
                acount += 1
                ad[a] = 1
            else:
                ad[a] += 1
        vd = [(k,v) for k,v in vd.items()]
        vd.sort(key = lambda x: x[1], reverse=True)
        ad = [(k,v) for k,v in ad.items()]
        ad.sort(key = lambda x: x[1], reverse=True)
        with open(ofile1, 'w') as fp:
            for line in vd:
                fp.write(self.id2verb[line[0]] + '\t' + str(line[1]) + '\n')
        with open(ofile2, 'w') as fp:
            for line in ad:
                fp.write(self.id2arg[line[0]] + '\t' + str(line[1]) + '\n')
        print ("#Verb in main event", vcount)
        print ("#Arg in main event", acount)
        return vcount, acount
    
    def dump_process_length_distribution(self, ofile1="./length_distribution.txt"):
        ld = {}
        for p in self.processes:
            l = len(p)
            if ld.get(l) is None:
                ld[l] = 1
            else:
                ld[l] += 1
        ld = [(k,v) for k,v in ld.items()]
        ld.sort(key = lambda x: x[0], reverse=False)
        with open(ofile1, 'w') as fp:
            fp.write('length\tcount')
            for line in ld:
                fp.write(str(line[0]) + '\t' + str(line[1]) + '\n')
    
    def dump_dataset_format(self, ofile='data_set.tsv', format='seq'):
        assert (format in ['seq', 'triple'])
        with open(ofile, 'w', encoding='utf-8') as fp:
            for i in range(len(self.processes)):
                if format == 'seq':
                    fp.write('\t'.join(self.processes[i]))
                    fp.write('\t' + self.id2verb[self.verb_id[i]] + '\t' + self.id2arg[self.arg_id[i]] + '\n')
                else:
                    for j in range(len(self.processes[i])):
                        fp.write(self.processes[i][j] + '\t' + self.id2verb[self.verb_id[i]] + '\t' + self.id2arg[self.arg_id[i]] + '\n')
    
    # Join with special cases special tokens
    def join_batch_sent(self, batch, begin='[CLS] ', sep=' [SEP] ', eos=''):
        return [begin+ sep.join(x) + eos for x in batch]
    
    # n-1 gram processes and true ids
    def minus_one_gram(self, batch, target, preserve=True):
        rt, ids = [], []
        assert (len(batch) == len(target))
        if preserve:
            rt, ids = [x for x in batch], [x for x in target]
        for i in range(len(batch)):
            for j in range(len(batch[i])):
                rt.append([x for k,x in enumerate(batch[i]) if k!=j])
                ids.append(target[i])
        return rt, ids
        

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