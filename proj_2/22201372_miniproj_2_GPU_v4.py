from typing import Any
import torch
import os
import h5py 
import numpy as np
import random
import itertools
from tqdm import tqdm
import pickle

"""
I only utilize torch for the sake of transferring data to GPU and perform matrix operations faster.
I do not utilize any automatic gradient operators of torch.
(All requires_grad are False, there are no parameters registered to torch.autograd, no torch's .backward() and .step() calls are present in the code).
All learning functions are written from scratch, and the matrix notations are compliant with the course content.
Bahri Batuhan Bilecen, 2023
"""

class HARDataset():
    def __init__(self, dataset_path):
        hf = h5py.File(dataset_path, 'r')

        self.trX = torch.from_numpy(np.array(hf['trX'])).to('cuda:0').type(torch.float32)
        self.trY = torch.from_numpy(np.array(hf['trY'])).to('cuda:0').type(torch.int8)
        self.tstX = torch.from_numpy(np.array(hf['tstX'])).to('cuda:0').type(torch.float32)
        self.tstY = torch.from_numpy(np.array(hf['tstY'])).to('cuda:0').type(torch.int8)

        # Extract the validation set and modify the training set
        val_indices = np.array(random.sample(range(1,500),300))
        for i in range(1,7):
            val_indices[50*(i-1):50*i] += 500*(i-1)
        self.valX = self.trX[val_indices]
        self.valY = self.trY[val_indices]
        mask = torch.ones(self.trX.size(0), dtype=torch.bool)
        mask[val_indices] = 0
        self.trX = self.trX[mask]
        self.trY = self.trY[mask]

        self.shuffle_ds()
        
    def shuffle_ds(self):
        indices = torch.randperm(self.trX.size(0))
        self.trX = self.trX[indices]
        self.trY = self.trY[indices]

class DataLoader():
    def __init__(self, dataset:HARDataset, type):
        self.type = type
        self.dataset = dataset

    def reload_ds(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        if self.type=='train':
            return self.dataset.trX[idx], self.dataset.trY[idx]
        elif self.type=='val':
            return self.dataset.valX[idx], self.dataset.valY[idx]
        elif self.type=='test':
            return self.dataset.tstX[idx], self.dataset.tstY[idx]

    def __len__(self):
        if self.type=='train':
            return self.dataset.trX.shape[0]
        elif self.type=='val':
            return self.dataset.valX.shape[0]
        elif self.type=='test':
            return self.dataset.tstX.shape[0]

class Net():
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate

        self.W_ih = 0.02 * torch.rand(size=(hidden_size, input_size+1), requires_grad=False, device='cuda:0') - 0.01
        self.W_hh = 0.02 * torch.rand(size=(hidden_size, hidden_size+1), requires_grad=False, device='cuda:0') - 0.01
        self.W_ho = 0.02 * torch.rand(size=(output_size, hidden_size+1), requires_grad=False, device='cuda:0') - 0.01

        self.cache = {}

    def extend_input(self, x):
        return torch.cat([x, -1*torch.ones((1,1), device='cuda:0', requires_grad=False)], axis=0)

    def forward(self, inputs, h_previous=None):
        h_states = []
        
        if h_previous is None:
            h_prev = torch.zeros((self.hidden_size, 1), device='cuda:0', requires_grad=False)
        else:
            h_prev = h_previous

        for i in range(len(inputs)):
            i_t = self.extend_input(inputs[[i]].T)
            h_t = torch.tanh(self.W_ih @ i_t + self.W_hh @ self.extend_input(h_prev))
            h_prev = h_t
            h_states.append(h_t)
        
        o_t = self.W_ho @ self.extend_input(h_t)
        y_t = torch.sigmoid(o_t)

        self.cache['h_states'] = h_states
        self.cache['inputs'] = inputs

        return y_t
    
    def backward(self, output, targets, h_previous=None):
        dWih = torch.zeros_like(self.W_ih)
        dWhh = torch.zeros_like(self.W_hh)
        dWho = torch.zeros_like(self.W_ho)
        
        d_t = targets[:,None]
        y_last = output
        h_last = self.cache['h_states'][-1]

        dy = y_last - d_t
        dWho = dy @ self.extend_input(h_last).T
        
        dh = (self.W_ho[:,:-1].T @ dy)

        for t in reversed(range(len(self.cache['inputs']))):
            h_t = self.cache['h_states'][t]
            i_t = self.cache['inputs'][[t]].T

            dh = (1-h_t**2) * dh # 1-h**2 is the tanh derivative
            
            if t>=1:
                dWhh += dh @ self.extend_input(self.cache['h_states'][t-1]).T
            else:
                if h_previous is None:
                    h_prev = torch.zeros((self.hidden_size+1,1), device='cuda:0', requires_grad=False).T
                else:
                    h_prev = self.extend_input(h_previous).T
                dWhh += dh @ h_prev
            
            dWih += dh @ self.extend_input(i_t).T

            dh = self.W_hh[:,:-1] @ dh

        # Gradient clipping to avoid exploding
        torch.clamp_(dWho, -0.5 , 0.5)
        torch.clamp_(dWih, -0.5 , 0.5)
        torch.clamp_(dWhh, -0.5 , 0.5)

        self.W_ih -= self.lr * dWih
        self.W_hh -= self.lr * dWhh
        self.W_ho -= self.lr * dWho
        
if __name__ == '__main__':
    dataset_path = 'data-Mini Project 2.h5'

    epoch_num = 50
    learning_rates = [0.05, 0.1]
    layer_sizes = [50, 100]
    batch_sizes = [10, 30]

    for comb in itertools.product(*[learning_rates, layer_sizes, batch_sizes]):
        lr, N, bs = comb
        config_name = f"{epoch_num}_{lr}_{N}_{bs}"
        os.makedirs(config_name, exist_ok=True)
        
        print(f'#######: {comb}')

        ds = HARDataset(dataset_path=dataset_path)
        train_dataloader = DataLoader(dataset=ds, type='train')
        test_dataloader = DataLoader(dataset=ds, type='test')
        val_dataloader = DataLoader(dataset=ds, type='val')

        net = Net(input_size=3, hidden_size=N, output_size=6, learning_rate=lr)

        # Training session
        pbar_train = tqdm(train_dataloader, desc='Training session')
        train_log = []
        val_log = []
        val_topk_log = []
        for curr_epoch in range(epoch_num):
            for idx, (data, label) in enumerate(pbar_train):
                data_chunks = data.split(bs) # split data for truncated BPTT 
                h_prev = None
                for d in data_chunks:
                    out = net.forward(d, h_prev)
                    E =  -(label * torch.log(out) + (1-label) * torch.log(1-out)).mean()
                    net.backward(out, label, h_prev)
                    h_prev = net.cache['h_states'][-1] # update the initial h condition for the next batch in truncated BPTT
                if idx % 5 == 0:
                    train_log.append([idx+curr_epoch*len(train_dataloader), E])
                if idx % 25 == 0: 
                    pbar_train.set_description(f"Training session. Epoch:{curr_epoch} E:{E}")

            # Validation session per epoch
            pbar_val = tqdm(val_dataloader, desc='Val session')
            val_topk_acc = [0]*6
            E_val = []
            for idx, (data, label) in enumerate(pbar_val):
                out = net.forward(data)
                E_val.append(-(label * torch.log(out) + (1-label) * torch.log(1-out)).mean())
                sorted_out_idx = torch.argsort(out.mean(dim=1), dim=0, descending=True)
                top_k = torch.argmax(label[sorted_out_idx])
                for i in range(top_k, 6):
                    val_topk_acc[i] += 1
            val_topk_acc = [x/len(val_dataloader) for x in val_topk_acc]
            E_val_mean = sum(E_val)/len(E_val)

            val_log.append([curr_epoch*len(train_dataloader), E_val_mean])
            val_topk_log.append(val_topk_acc)
            
            print(f"Val top-k accuracy scores: {val_topk_acc} for epoch {curr_epoch}")
            print(f"Val error: {E_val_mean}")

            # Shuffle dataset after an epoch
            ds.shuffle_ds()
            train_dataloader.reload_ds(ds)
            pbar_train = tqdm(train_dataloader, desc='Training session')
            
            # Save ckpt
            with open(os.path.join(config_name, f"{curr_epoch}_{config_name}_valacc:{val_topk_acc}_trainE:{E}.NET"), "wb") as f:
                pickle.dump(net, f, pickle.HIGHEST_PROTOCOL)

        # Testing session at the end
        pbar = tqdm(test_dataloader, desc='Testing session')
        topk_acc = [0]*6
        for idx, (data, label) in enumerate(pbar):
            out = net.forward(data)
            sorted_out_idx = torch.argsort(out.mean(dim=1), dim=0, descending=True)
            top_k = torch.argmax(label[sorted_out_idx])
            for i in range(top_k, 6):
                topk_acc[i] += 1
        topk_acc = [x/len(test_dataloader) for x in topk_acc]
        print(f"Test top-k accuracy scores: {topk_acc}")

        # Save ckpt and logs for reports
        with open(os.path.join(config_name, f"{curr_epoch}_{config_name}_testacc:{topk_acc}_trainE:{E}.NET"), "wb") as f:
            pickle.dump(net, f, pickle.HIGHEST_PROTOCOL)
        t_log = np.array([[tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor for tensor in sublist] for sublist in train_log])
        v_log = np.array([[tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor for tensor in sublist] for sublist in val_log])
        v_topk_log = np.array([[tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor for tensor in sublist] for sublist in val_topk_log])
        np.save(os.path.join(config_name,f"{curr_epoch}_{config_name}_test_acc:{topk_acc}.npy"),t_log.T)
        np.save(os.path.join(config_name,f"{curr_epoch}_{config_name}_val_error:{E_val_mean}.npy"),v_log.T)
        np.save(os.path.join(config_name,f"{curr_epoch}_{config_name}_val_acc:{val_topk_acc}.npy"),v_topk_log.T)
