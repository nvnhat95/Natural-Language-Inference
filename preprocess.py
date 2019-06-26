#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create the data for sentence pair classification
"""

import os, sys, glob, re
import argparse
import numpy as np
import h5py
import itertools
from collections import defaultdict

class Indexer:
    def __init__(self, symbols = ["<blank>","<unk>","<s>","</s>"]):
        self.vocab = defaultdict(int)
        self.PAD = symbols[0]
        self.UNK = symbols[1]
        self.BOS = symbols[2]
        self.EOS = symbols[3]
        self.d = {self.PAD: 1, self.UNK: 2, self.BOS: 3, self.EOS: 4}

    def add_w(self, ws):
        for w in ws:
            if w not in self.d:
                self.d[w] = len(self.d) + 1
            
    def convert(self, w):        
        return self.d[w] if w in self.d else self.d['<oov' + str(np.random.randint(1,100)) + '>']

    def convert_sequence(self, ls):
        return [self.convert(l) for l in ls]

    def clean(self, s):
        s = s.replace(self.PAD, "")
        s = s.replace(self.BOS, "")
        s = s.replace(self.EOS, "")
        return s
        
    def write(self, outfile):
        out = open(outfile, "w")
        items = [(v, k) for k, v in self.d.items()]
        items.sort()
        for v, k in items:
            out.write("{} {}\n".format(k, v))
        out.close()

    def prune_vocab(self, k, cnt=False):
        vocab_list = [(word, count) for word, count in self.vocab.items()]        
        if cnt:
            self.pruned_vocab = {pair[0]:pair[1] for pair in vocab_list if pair[1] > k}
        else:
            vocab_list.sort(key = lambda x: x[1], reverse=True)
            k = min(k, len(vocab_list))
            self.pruned_vocab = {pair[0]:pair[1] for pair in vocab_list[:k]}
        for word in self.pruned_vocab:
            if word not in self.d:
                self.d[word] = len(self.d) + 1

    def load_vocab(self, vocab_file):
        self.d = {}
        for line in open(vocab_file, 'r'):
            v, k = line.strip().split()                
            self.d[v] = int(k)
            
def pad(ls, length, symbol, pad_back = True):
    if len(ls) >= length:
        return ls[:length]
    if pad_back:
        return ls + [symbol] * (length -len(ls))
    else:
        return [symbol] * (length -len(ls)) + ls        

def get_glove_words(f):
    glove_words = set()
    for line in open(f, "r", encoding='utf-8'):
        word = line.split()[0].strip()
        glove_words.add(word)
    return glove_words

def get_data(args):
    word_indexer = Indexer(["<blank>","<unk>","<s>","</s>"])
    label_indexer = Indexer(["<blank>","<unk>","<s>","</s>"])
    POS_indexer = Indexer(["<blank>","<unk>","<s>","</s>"])
    label_indexer.d = {}
    POS_indexer.d = {}
    POS_indexer.vocab["BOS"] = 100000
    glove_vocab = get_glove_words(args.glove)
    for i in range(1,101): #hash oov words to one of 100 random embeddings, per Parikh et al. 2016
        oov_word = '<oov'+ str(i) + '>'
        word_indexer.vocab[oov_word] += 1
    def make_vocab(srcfile, targetfile, labelfile, seqlength):
        num_sents = 0
        for _, (src_orig, targ_orig, label_orig) in \
                enumerate(zip(open(srcfile,'r'), open(targetfile,'r'), open(labelfile, 'r'))):
            src_orig = word_indexer.clean(src_orig.strip())
            targ_orig = word_indexer.clean(targ_orig.strip())                
            targ, targ_POS = targ_orig.strip().split('\t')
            targ = targ.strip().split()
            targ_POS = targ_POS.strip().split()
            src, src_POS = src_orig.strip().split('\t')
            src = src.strip().split()
            src_POS = src_POS.strip().split()
            label = label_orig.strip().split()
            
            if len(targ) > seqlength or len(src) > seqlength or len(targ) < 1 or len(src) < 1:
                continue
            num_sents += 1
            for word in targ:
                if word in glove_vocab:
                    word_indexer.vocab[word] += 1

            for pos in targ_POS:
                POS_indexer.vocab[pos] += 1
                    
            for word in src:
                if word in glove_vocab:
                    word_indexer.vocab[word] += 1
                    
            for pos in src_POS:
                POS_indexer.vocab[pos] += 1
                        
            for word in label:
                label_indexer.vocab[word] += 1
                
        return num_sents
                
    def convert(srcfile, targetfile, labelfile, batchsize, seqlength, outfile, num_sents,
                max_sent_l=0, shuffle=0):
        
        newseqlength = seqlength + 1 #add 1 for BOS
        targets = np.zeros((num_sents, newseqlength), dtype=int)
        targets_POS = np.zeros((num_sents, newseqlength), dtype=int)
        sources = np.zeros((num_sents, newseqlength), dtype=int)
        sources_POS = np.zeros((num_sents, newseqlength), dtype=int)
        labels = np.zeros((num_sents,), dtype =int)
        source_lengths = np.zeros((num_sents,), dtype=int)
        target_lengths = np.zeros((num_sents,), dtype=int)
        both_lengths = np.zeros(num_sents, dtype = {'names': ['x','y'], 'formats': ['i4', 'i4']})
        dropped = 0
        sent_id = 0
        for _, (src_orig, targ_orig, label_orig) in \
                enumerate(zip(open(srcfile,'r'), open(targetfile,'r')
                                         ,open(labelfile,'r'))):
            src_orig = word_indexer.clean(src_orig.strip())
            targ_orig = word_indexer.clean(targ_orig.strip())
            
            targ, targ_POS = targ_orig.strip().split('\t')
            targ = targ.strip().split()
            targ_POS = targ_POS.strip().split()
            src, src_POS = src_orig.strip().split('\t')
            src = src.strip().split()
            src_POS = src_POS.strip().split()
            
            targ =  [word_indexer.BOS] + targ
            src =  [word_indexer.BOS] + src
            targ_POS = ["BOS"] + targ_POS
            src_POS = ["BOS"] + src_POS
            
            label = label_orig.strip().split()
            max_sent_l = max(len(targ), len(src), max_sent_l)
            if len(targ) > newseqlength or len(src) > newseqlength or len(targ) < 2 or len(src) < 2:
                dropped += 1
                continue                   
            targ = pad(targ, newseqlength, word_indexer.PAD)
            targ = word_indexer.convert_sequence(targ)
            targ = np.array(targ, dtype=int)

            targ_POS = pad(targ_POS, newseqlength, "BOS")
            targ_POS = POS_indexer.convert_sequence(targ_POS)
            targ_POS = np.array(targ_POS, dtype=int)
            
            src = pad(src, newseqlength, word_indexer.PAD)
            src = word_indexer.convert_sequence(src)
            src = np.array(src, dtype=int)
            
            src_POS = pad(src_POS, newseqlength, "BOS")
            src_POS = POS_indexer.convert_sequence(src_POS)
            src_POS = np.array(src_POS, dtype=int)
            
            targets[sent_id] = np.array(targ,dtype=int)
            targets_POS[sent_id] = np.array(targ_POS,dtype=int)
            target_lengths[sent_id] = (targets[sent_id] != 1).sum()
            sources[sent_id] = np.array(src, dtype=int)
            sources_POS[sent_id] = np.array(src_POS,dtype=int)
            source_lengths[sent_id] = (sources[sent_id] != 1).sum()            
            labels[sent_id] = label_indexer.d[label[0]]
            both_lengths[sent_id] = (source_lengths[sent_id], target_lengths[sent_id])
            sent_id += 1
            if sent_id % 100000 == 0:
                print("{}/{} sentences processed".format(sent_id, num_sents))

        print(sent_id, num_sents)
        if shuffle == 1:
            rand_idx = np.random.permutation(sent_id)
            targets = targets[rand_idx]
            sources = sources[rand_idx]
            source_lengths = source_lengths[rand_idx]
            target_lengths = target_lengths[rand_idx]
            labels = labels[rand_idx]
            both_lengths = both_lengths[rand_idx]
        
        #break up batches based on source/target lengths
        
        
        source_lengths = source_lengths[:sent_id]
        source_sort = np.argsort(source_lengths) 

        both_lengths = both_lengths[:sent_id]
        sorted_lengths = np.argsort(both_lengths, order = ('x', 'y'))        
        sources = sources[sorted_lengths]
        targets = targets[sorted_lengths]
        labels = labels[sorted_lengths]
        target_l = target_lengths[sorted_lengths]
        source_l = source_lengths[sorted_lengths]

        curr_l_src = 0
        curr_l_targ = 0
        l_location = [] #idx where sent length changes
        
        for j,i in enumerate(sorted_lengths):
            if source_lengths[i] > curr_l_src or target_lengths[i] > curr_l_targ:
                curr_l_src = source_lengths[i]
                curr_l_targ = target_lengths[i]
                l_location.append(j+1)
        l_location.append(len(sources))

        #get batch sizes
        curr_idx = 1
        batch_idx = [1]
        batch_l = []
        target_l_new = []
        source_l_new = []
        for i in range(len(l_location)-1):
            while curr_idx < l_location[i+1]:
                curr_idx = min(curr_idx + batchsize, l_location[i+1])
                batch_idx.append(curr_idx)
        for i in range(len(batch_idx)-1):
            batch_l.append(batch_idx[i+1] - batch_idx[i])
            source_l_new.append(source_l[batch_idx[i]-1])
            target_l_new.append(target_l[batch_idx[i]-1])
        # Write output
        f = h5py.File(outfile, "w")        
        f["source"] = sources
        f["source_POS"] = sources_POS
        f["target"] = targets
        f["target_POS"] = targets_POS
        f["target_l"] = np.array(target_l_new, dtype=int)
        f["source_l"] = np.array(source_l_new, dtype=int)
        f["label"] = np.array(labels, dtype=int)
        f["label_size"] = np.array([len(np.unique(np.array(labels, dtype=int)))])
        f["batch_l"] = np.array(batch_l, dtype=int)
        f["batch_idx"] = np.array(batch_idx[:-1], dtype=int)
        f["source_size"] = np.array([len(word_indexer.d)])
        f["target_size"] = np.array([len(word_indexer.d)])
        f["POS_size"] = len(POS_indexer.d)
        print("Saved {} sentences (dropped {} due to length/unk filter)".format(
            len(f["source"]), dropped))
        f.close()                
        return max_sent_l

    print("First pass through data to get vocab...")
    num_sents_train = make_vocab(args.premise_train, args.hypothesis_train, args.label_train,
                                             args.seqlength)
    print("Number of sentences in training: {}".format(num_sents_train))
    num_sents_valid = make_vocab(args.premise_val, args.hypothesis_val, args.label_val,
                                             args.seqlength)
    print("Number of sentences in valid: {}".format(num_sents_valid))
    num_sents_test = make_vocab(args.premise_test, args.hypothesis_test, args.label_test,
                                             args.seqlength)
    print("Number of sentences in test: {}".format(num_sents_test))    
    
    #prune and write vocab
    word_indexer.prune_vocab(0, True)
    label_indexer.prune_vocab(1000) 
    POS_indexer.prune_vocab(0, True)

    word_indexer.write(os.path.join(args.out_folder, "word.dict"))
    label_indexer.write(os.path.join(args.out_folder, "label.dict"))
    POS_indexer.write(os.path.join(args.out_folder, "POS.dict"))
    
    print("Source vocab size: Original = {}, Pruned = {}".format(len(word_indexer.vocab), 
                                                          len(word_indexer.d)))
    print("Target vocab size: Original = {}, Pruned = {}".format(len(word_indexer.vocab), 
                                                          len(word_indexer.d)))

    max_sent_l = 0
    max_sent_l = convert(args.premise_val, args.hypothesis_val, args.label_val,
                         args.batchsize, args.seqlength,
                         os.path.join(args.out_folder, "dev.hdf5"), num_sents_valid,
                         max_sent_l)
    max_sent_l = convert(args.premise_train, args.hypothesis_train, args.label_train,
                         args.batchsize, args.seqlength,
                         os.path.join(args.out_folder, "train.hdf5"), num_sents_train, 
                         max_sent_l)
    max_sent_l = convert(args.premise_test, args.hypothesis_test, args.label_test,
                         args.batchsize, args.seqlength,
                         os.path.join(args.out_folder, "test.hdf5"), num_sents_test, 
                         max_sent_l)    
    print("Max sent length (before dropping): {}".format(max_sent_l))    

    
def parse_data(args):
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    file_names = {}
    file_names['train'] = glob.glob(args.data_folder + '/*_train.txt')[0]
    file_names['dev'] = glob.glob(args.data_folder + '/*_dev.txt')[0]
    file_names['test'] = glob.glob(args.data_folder + '/*_test.txt')[0]

    for split in ["train", "dev", "test"]:
        src_out = open(os.path.join(args.out_folder, "premise_" + split + ".txt"), "w")
        targ_out = open(os.path.join(args.out_folder, "hypothesis_" + split + ".txt"), "w")
        label_out = open(os.path.join(args.out_folder, "label_" + split + ".txt"), "w")
        label_set = set(["neutral", "entailment", "contradiction"])

        for line in open(file_names[split], "r"):
            d = line.split("\t")
            label = d[0].strip()
            premise = " ".join(d[1].replace("(", "").replace(")", "").strip().split())
            premise_POS = " ".join(re.findall("\(([^\()]+) [^\()]+\)", d[3]))
            hypothesis = " ".join(d[2].replace("(", "").replace(")", "").strip().split())
            hypothesis_POS = " ".join(re.findall("\(([^\()]+) [^\()]+\)", d[4]))
            if args.lowercase:
                premise = premise.lower()
                hypothesis = hypothesis.lower()
            if label in label_set:
                src_out.write(premise + "\t" + premise_POS + "\n")
                targ_out.write(hypothesis + "\t" + hypothesis_POS + "\n")
                label_out.write(label + "\n")

        src_out.close()
        targ_out.close()
        label_out.close()

def load_glove_vec(args):
    vocab = {}
    with open(os.path.join(args.out_folder, 'word.dict'), "r") as f:
        for line in f:
            line = line.strip('\n').split(' ')
            vocab[line[0]] = int(line[1])
    
    len_vocab = len(vocab)
    print("vocab size is {}".format(len_vocab))
    
    with open(args.glove, 'r', encoding='utf-8') as f:
        glove_dim = len(f.readline().split(' ')) - 1
    print("word embedding dimension: {}".format(glove_dim))
    w2v_vecs = np.random.normal(size = (len_vocab, glove_dim))
    
    w2v = {}
    for line in open(args.glove, 'r', encoding='utf-8'):
        d = line.split()
        word = d[0]
        vec = np.array([float(x) for x in d[1:]])
        if word in vocab:
            w2v[word] = vec
    
    print("num words in pretrained model is " + str(len(w2v)))
    for word in w2v:
        w2v_vecs[vocab[word] - 1 ] = w2v[word]
    for i in range(len(w2v_vecs)):
        w2v_vecs[i] = w2v_vecs[i] / np.linalg.norm(w2v_vecs[i])
    with h5py.File(os.path.join(args.out_folder, 'glove.hdf5'), "w") as f:
        f["word_vecs"] = np.array(w2v_vecs, dtype=np.float32)

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
                                                
    parser.add_argument('--data_folder', help="location of folder with the snli files")
    parser.add_argument('--out_folder', help="location of the output folder", default='preprocessed')
    
    parser.add_argument('--vocabsize', help="Size of source vocabulary, constructed "
                                                "by taking the top X most frequent words. "
                                                " Rest are replaced with special UNK tokens.",
                                                type=int, default=50000)
    
    parser.add_argument('--batchsize', help="Size of each minibatch.", type=int, default=32)
    parser.add_argument('--lowercase', help="convert all word to lowercase.", type=bool, default=True)
    parser.add_argument('--seqlength', help="Maximum sequence length. Sequences longer "
                                               "than this are dropped.", type=int, default=100)

    parser.add_argument('--glove', type = str, default = '')    
    args = parser.parse_args()
    args.premise_train = os.path.join(args.out_folder, 'premise_train.txt')
    args.hypothesis_train = os.path.join(args.out_folder, 'hypothesis_train.txt')
    args.label_train = os.path.join(args.out_folder, 'label_train.txt')
    args.premise_val = os.path.join(args.out_folder, 'premise_dev.txt')
    args.hypothesis_val = os.path.join(args.out_folder, 'hypothesis_dev.txt')
    args.label_val = os.path.join(args.out_folder, 'label_dev.txt')
    args.premise_test = os.path.join(args.out_folder, 'premise_test.txt')
    args.hypothesis_test = os.path.join(args.out_folder, 'hypothesis_test.txt')
    args.label_test = os.path.join(args.out_folder, 'label_test.txt')
    
    parse_data(args)
    
    get_data(args)
    
    load_glove_vec(args)