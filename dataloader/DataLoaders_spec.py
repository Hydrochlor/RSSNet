import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F
import random
import numpy as np
import time
import os

def make_dataloader(is_train = True, data_kwargs = None, num_workers = 4,
                    batch_size = 16, upsample = True):
    dataset = Datasets(**data_kwargs)
    return DataLoaders(dataset, is_train = is_train, 
                        batch_size = batch_size, num_workers = num_workers, up = upsample)

class Datasets(Dataset):
    def __init__(self, ref_path = None, list_path = None, mix_path = None):
        super(Datasets, self).__init__()
        origin_spec = np.loadtxt(ref_path)
        spec_list = np.loadtxt(list_path)
        self.mix_spec = np.loadtxt(mix_path)
        self.ref_spec = []
        for i in range(len(spec_list)):
            label1 = int(spec_list[i][0])
            label2 = int(spec_list[i][1])
            factor = spec_list[i][2]
            self.ref_spec.append([origin_spec[label1] * factor, origin_spec[label2] * (1 - factor)])
        self.ref_spec = np.array(self.ref_spec)

    def __len__(self):
        return len(self.mix_spec)

    def __getitem__(self, index):
        mix = self.mix_spec[index]
        ref = [r for r in self.ref_spec[index]]
        return {
            'mix': mix,
            'ref': ref
        }

class DataLoaders():
    def __init__(self, dataset, num_workers = 4, batch_size = 1, leng = 32000, is_train = True, up = True):
        super(DataLoaders, self).__init__()
        self.dataset = dataset
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.is_train = is_train
        self.leng = leng
        self.up = up
        self.data_loader = DataLoader(self.dataset,
                                      num_workers=self.num_workers,
                                      batch_size=self.batch_size // 2,
                                      #batch_size=self.batch_size,
                                      shuffle=self.is_train,
                                      collate_fn=self._collate
                                      )

    def _collate(self, batch):
        batch_spec = []
        for b in batch:
            batch_spec.append(b)
        return batch_spec

    def __len__(self):
        return len(self.data_loader) * (self.batch_size // 2)

    def upsample(self, batch):
        new_batch = {}
        new_batch['mix'] = F.interpolate(batch['mix'], self.leng)
        ref = []
        ref.append(F.interpolate(batch['ref'][0], self.leng))
        ref.append(F.interpolate(batch['ref'][1], self.leng))
        new_batch['ref'] = ref
        return new_batch

    def __iter__(self):
        mini_batch = []
        for batch in self.data_loader:
            #print(batch)
            mini_batch += batch
            length = len(mini_batch)
            if self.is_train:
                random.shuffle(mini_batch)
            collate_chunk = []
            for start in range(0, length-self.batch_size+1, self.batch_size):
                b = default_collate(
                    mini_batch[start:start+self.batch_size])
                collate_chunk.append(b)
            idx = length % self.batch_size
            mini_batch = mini_batch[-idx:] if idx else []
            for m_batch in collate_chunk:
                if self.up is True:
                    res = self.upsample(m_batch)
                else:
                    res = m_batch
                # res['mix'] = res['mix'].squeeze()
                # res['ref'][0] = res['ref'][0].squeeze()
                # res['ref'][1] = res['ref'][1].squeeze()
                yield res

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)    
    torch.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed)    
    np.random.seed(seed)    
    random.seed(seed)    
    torch.backends.cudnn.benchmark = False    
    torch.backends.cudnn.deterministic = True

import matplotlib.pyplot as plt
def draw(specs, title, savepath):
    plt.style.use('seaborn-whitegrid')
    figure = plt.figure()
    plt.suptitle(title)
    ax1 = figure.add_subplot(3,1,1)
    x1 = np.arange(0, len(specs[0]), 1)
    ax1.plot(x1, specs[0])

    ax2 = figure.add_subplot(3,1,2)
    x2 = np.arange(0, len(specs[1]), 1)
    ax2.plot(x2, specs[1])

    ax3 = figure.add_subplot(3,1,3)
    x3 = np.arange(0, len(specs[2]), 1)
    ax3.plot(x3, specs[2])

    plt.savefig(savepath)
    figure.clear()
    ax1.cla()
    ax2.cla()
    ax3.cla()

if __name__ == "__main__":
    setup_seed(42)
    ref_path = '/home/ubuntu/RSSNet/dataset/RRUFF-2Mix/prep_spec.txt'
    list_path = '/home/ubuntu/RSSNet/dataset/RRUFF-2Mix/train/mix_2_spec_train_60000.txt'  
    mix_path = '/home/ubuntu/RSSNet/dataset/RRUFF-2Mix/train/RRUFF_rednoise_10-20_mix_train.txt'
    d = Datasets(ref_path=os.path.join(os.path.dirname(__file__), ref_path),
                 list_path=os.path.join(os.path.dirname(__file__), list_path),
                 mix_path = os.path.join(os.path.dirname(__file__), mix_path))
    #dl = DataLoader(d,num_workers=4, batch_size=8, shuffle = True, collate_fn=_collate)
    dl = DataLoaders(d, 4, 16, 32000, True, False)
    print('here')
    for eg in dl:
        print(eg['mix'].shape)
        #draw(eg['mix'][0])
        print(eg['ref'][0].shape)
        print(eg['ref'][1].shape)
        break