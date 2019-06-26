import torch
import numpy as np
import h5py

class SNLI_data(object):
    '''
        class to handle training data
    '''

    def __init__(self, fname, use_padding=True, batch_size=32):
        self.use_padding = use_padding
        self.batch_size = batch_size

        f = h5py.File(fname, 'r')
        self.source = torch.from_numpy(np.array(f['source'])) - 1
        self.source_POS = torch.from_numpy(np.array(f['source_POS'])) - 1
        self.target = torch.from_numpy(np.array(f['target'])) - 1
        self.target_POS = torch.from_numpy(np.array(f['target_POS'])) - 1
        self.POS_size = f['POS_size']
        
        self.label = torch.from_numpy(np.array(f['label'])) - 1
        self.label_size = torch.from_numpy(np.array(f['label_size']))
        self.source_l = torch.from_numpy(np.array(f['source_l']))
        self.target_l = torch.from_numpy(np.array(f['target_l'])) # max target length each batch
        # idx in torch style; indicate the start index of each batch (starting
        # with 1)
        self.batch_idx = torch.from_numpy(np.array(f['batch_idx'])) - 1
        self.batch_l = torch.from_numpy(np.array(f['batch_l']))

        self.batches = []   # batches

        self.length = self.batch_l.size(0)  # number of batches

        self.size = 0   # number of sentences

        
    
    def get_batches(self):
        if self.use_padding:
            self.batches = []
            num_samples = self.source.shape[0]
            idx = np.arange(num_samples)
            np.random.shuffle(idx)
            self.source = self.source[idx]
            self.target = self.target[idx]
            self.label = self.label[idx]
            for i in range(0, num_samples - self.batch_size, self.batch_size):
                batch = (self.source[i : i + self.batch_size].type(torch.LongTensor),
                           self.target[i : i + self.batch_size].type(torch.LongTensor),
                           self.label[i : i + self.batch_size].type(torch.LongTensor))
                self.batches.append(batch)
        else:
            if self.batches == []:
                for i in range(self.length):
                    batch = (self.source[self.batch_idx[i] : self.batch_idx[i] + self.batch_l[i]][:, :self.source_l[i]].type(torch.LongTensor),
                           self.target[self.batch_idx[i] : self.batch_idx[i] + self.batch_l[i]][:, :self.target_l[i]].type(torch.LongTensor),
                           self.label[self.batch_idx[i] : self.batch_idx[i] + self.batch_l[i]].type(torch.LongTensor),
                           self.source_POS[self.batch_idx[i] : self.batch_idx[i] + self.batch_l[i]][:, :self.source_l[i]].type(torch.LongTensor),
                           self.target_POS[self.batch_idx[i] : self.batch_idx[i] + self.batch_l[i]][:, :self.target_l[i]].type(torch.LongTensor))
                    self.batches.append(batch)
                    self.size += self.batch_l[i]
        return self.batches