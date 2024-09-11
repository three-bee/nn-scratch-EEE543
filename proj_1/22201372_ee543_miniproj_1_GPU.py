import numpy as np
import os
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import pickle
import itertools
import gzip
import torch

class MNISTDataset():
    def __init__(self, dataset_path='train-images-idx3-ubyte.gz', label_path='train-labels-idx1-ubyte.gz', 
                 shuffle=False, one_hot_values=[0,1]):

        with gzip.open(dataset_path) as f:
            self.pixel_data = np.frombuffer(f.read(), "B", offset=16).astype('float32')
            self.pixel_data = self.pixel_data.reshape(-1,784) / 255.
            
        with gzip.open(label_path) as f:
            self.label_data = np.frombuffer(f.read(), "B", offset=8)
        
        # Either [0,1] or [-1,1], where the 1st & 2nd elements define negative & positive samples, respectively
        self.one_hot_values = one_hot_values
        if self.one_hot_values == [0,1]:
            self.label_data = np.eye(10)[self.label_data]
        elif self.one_hot_values == [-1,1]:
            self.label_data = 2 * np.eye(10)[self.label_data] - 1

        self.pixel_data = torch.from_numpy(self.pixel_data).to(device='cuda').float()
        self.label_data = torch.from_numpy(self.label_data).to(device='cuda').float()

        if shuffle:
            self.shuffle_dataset()

    def shuffle_dataset(self):
        indices = torch.randperm(len(self.label_data))
        self.pixel_data = self.pixel_data[indices]
        self.label_data = self.label_data[indices]

    def __getitem__(self, idx):
        """
        Reads an image from the dataset and returns along with the one-hot encoded class label. 
        """
        img = self.pixel_data[idx]
        one_hot_label = self.label_data[idx]
        return img, one_hot_label
    
    def __len__(self):
        return len(self.label_data)

class LinearLayer():
    def __init__(self, in_feat, out_feat, act_type='sigmoid', lr=0.01):
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.act_type = act_type

        self._w_ext = torch.normal(mean=0.0, std=0.01, size=(out_feat, in_feat+1), requires_grad=False, device='cuda:0') #0.02 * torch.rand(size=(out_feat, in_feat+1), requires_grad=False, device='cuda:0') - 0.01
        #torch.normal(mean=0.0, std=0.01, size=(out_feat, in_feat+1), requires_grad=False, device='cuda:0')
        self._w_ext[:,-1] = 0

        self._grad = torch.zeros((out_feat,1), requires_grad=False, device='cuda:0')
        self._gamma_prime = torch.eye(out_feat, out_feat, requires_grad=False, device='cuda:0')
        self._delta_w_list = []
        self._input = None
        self._minus_one = torch.ones(1, device='cuda:0', requires_grad=False)*-1
            
        self._eta = lr # Learning rate

    def _act(self, v):
        if self.act_type == 'relu':
            return torch.max(torch.tensor(0.0, device='cuda:0', requires_grad=False), v)
        elif self.act_type == 'sigmoid':
            return 1/(1+torch.exp(-v))
        elif self.act_type == 'tanh':
            return torch.tanh(v)
            # NOTE Using exponentials yield very large numbers without any clip operations, resulting in NaN's.
            #a = torch.exp(v)
            #b = torch.exp(-v)
            #return (a-b)/(a+b)
    
    def _act_prime(self, o): # Closed-form of the activation derivatives
        if self.act_type == 'relu':  
            return torch.where(o>=0, torch.tensor(1.0, device='cuda:0', requires_grad=False), torch.tensor(0.0, device='cuda:0', requires_grad=False))
        elif self.act_type == 'sigmoid':
            return o*(1-o)
        elif self.act_type == 'tanh':
            return (1-o**2)

    def forward(self, x):
        """
        Returns the matrix multiplication of extended weight matrix (_w_ext) with the extended input (_input) as the output.
        o(out_feat, 1) = _w_ext(out_feat x in_feat+1) @ _input(in_feat+1 x 1)
        Meanwhile, records the derivative of output activations (_gamma_prime(out_feat x out_feat) = diag(_act_prime(out_feat, 1))).
        """
        self._input = torch.cat([x,self._minus_one])
        v = self._w_ext @ self._input
        o = self._act(v)
        if self.act_type == 'relu':
            self._gamma_prime = torch.diag(self._act_prime(v))
        else:
            self._gamma_prime = torch.diag(self._act_prime(o))
        return o
    
    def backward(self, prev_grad, prev_weights):
        """
        Calculates the accumulated gradient for the current layer by multiplying output activation derivatives, previous weights and previous accumulated gradients.
        _grad(out_feat x 1) = _gamma_prime(out_feat x out_feat) @ prev_weights.T(out_feat x in_feat) @ prev_grad(in_feat x 1)

        Gradient step direction for updating _w_ext is also calculated in here as delta_w. In case of batch learning, delta_w is recorded for later use.
        _w_ext (out_feat x in_feat+1) += _eta * _grad(out_feat x 1) @ _input.T(1 x in_feat+1)
        """
        self._grad = self._gamma_prime @ prev_weights.T @ prev_grad 
        delta_w = self._eta * self._grad[:,None] @ self._input[:,None].T
        # Only the weight update part changes in the L2 weight regularization part.
        # Taking the partial derivative of the square-sum of all weight terms with respect to a specific weight only yields that weight.
        if l2_regularizer_lambda > 0.0: 
            delta_w -= l2_regularizer_lambda * self._w_ext
        self._delta_w_list.append(delta_w)

    def step(self):
        """
        Updates the extended weight matrix by stepping towards the gradient descent direction.
        In case of batch learning, the step is the accumulation of the recorded steps in _delta_w_list.
        """
        self._w_ext += (sum(self._delta_w_list)/(len(self._delta_w_list)))
        self._delta_w_list = []

class Net():
    """
    A basic wrapper class for the neural network.
    """
    def __init__(self, layer_nums=[[28*28,300],[300,10]], acts=['relu', 'sigmoid'], learning_rate=0.01):
        self.layer_nums = layer_nums
        self.out_num = layer_nums[-1][-1]
        self.perceptron = []
        self.eye = torch.eye(self.out_num, self.out_num, device='cuda:0', requires_grad=False)

        assert len(layer_nums) == len(acts)
        for i in range(len(layer_nums)): # Construct the perceptron layers
            self.perceptron.append(LinearLayer(in_feat=layer_nums[i][0], out_feat=layer_nums[i][1], act_type=acts[i], lr=learning_rate))

    def forward(self, x):
        x_ = x
        for i in range(len(self.perceptron)):
            x_ = self.perceptron[i].forward(x_)
        return x_
    
    def backward(self, error):
        for i in reversed(range(len(self.perceptron))):
            if i == len(self.perceptron)-1: # Last layer, behave as if there is an another layer with fixed identity weights
                self.perceptron[i].backward(prev_grad=error, 
                                            prev_weights=self.eye)
            else:
                self.perceptron[i].backward(prev_grad=self.perceptron[i+1]._grad, 
                                            prev_weights=self.perceptron[i+1]._w_ext[:,:-1]) # Omit the bias from the extended weight matrix
        
    def step(self):
        for i in range(len(self.perceptron)):
            self.perceptron[i].step()

if __name__ == '__main__':
    """
    IMPORTANT: 
    I only utilize torch for the sake of transferring data to GPU and perform matrix operations faster. Using numpy and training on CPU is awfully slow.
    I do not utilize any automatic gradient operators of torch.
    (All requires_grad are False, there are no parameters registered to torch.autograd, no torch's .backward() and .step() calls are present in the code).
    All learning functions are written from scratch, and the matrix notations are compliant with the course content.
    
    Bahri Batuhan Bilecen, 2023
    """
    # Fix pseudo-random number generator seeds for reproducability and to keep our sanity in tact
    random.seed(1071)
    os.environ['PYTHONHASHSEED'] = str(1453)
    np.random.seed(1881)
    torch.manual_seed(1920)
    torch.cuda.manual_seed(1923)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Hyperparameter combinations
    h = [300,500,1000]
    l = [0.01,0.05,0.09]
    a = [['relu','sigmoid'],['tanh','tanh']]
    b = [10, 50, 100]
    r = [0.0, 0.01, 0.001]
    for comb in itertools.product(*[h,l,a,b,r]):
        epoch_num = 50
        hidden_neuron_num = comb[0]
        lr = comb[1]
        acts = comb[2]
        batch_size = comb[3]
        l2_regularizer_lambda = comb[4]

        if acts[-1] == 'sigmoid':
            one_hot_values = [0,1]
        elif acts[-1] == 'tanh':
            one_hot_values = [-1,1]
        
        config_name = f"bs:{batch_size}_ep:{epoch_num}_hidden:{hidden_neuron_num}_lr:{lr}_act:{acts[0]}-{acts[1]}_l2:{l2_regularizer_lambda}"
        print(f'#### Config name: {config_name}')

        train_set = MNISTDataset(dataset_path='train-images-idx3-ubyte.gz', label_path='train-labels-idx1-ubyte.gz',
                                shuffle=True, one_hot_values=one_hot_values)
        test_set = MNISTDataset(dataset_path='t10k-images-idx3-ubyte.gz', label_path='t10k-labels-idx1-ubyte.gz',
                                shuffle=False, one_hot_values=one_hot_values)
        network = Net(layer_nums=[[28*28,hidden_neuron_num], [hidden_neuron_num,10]], acts=acts, learning_rate=lr)

        # Start training session
        pbar = tqdm(train_set, desc='Training the MNIST classifier perceptron:')
        train_log = []
        for curr_epoch in range(epoch_num):
            for idx, (x, d) in enumerate(pbar):
                o = network.forward(x)
                e = d - o
                network.backward(e)
                if idx % batch_size == 0: # accumulate grads of size=batch_size, then update weights
                    network.step()
                E = torch.square(e).mean()
                if l2_regularizer_lambda > 0.0:
                    E += 0.5 * l2_regularizer_lambda * sum([torch.sum(torch.square(network.perceptron[i]._w_ext)) for i in range(len(network.layer_nums))])
                if idx % 10_000 == 0:
                    pbar.set_description(f"Training the MNIST classifier perceptron. Epoch:{curr_epoch} E:{E}")
                    train_log.append([idx+curr_epoch*len(train_set), E])
            train_set.shuffle_dataset()
            pbar = tqdm(train_set, desc='Training the MNIST classifier perceptron:')
        
        # Start testing session
        pbar = tqdm(test_set, desc='Testing the MNIST classifier perceptron:')
        accuracy = 0
        for idx, (x, d) in enumerate(pbar):
            o = network.forward(x)
            if d[o.argmax()] == 1:
                accuracy +=1
        total_accuracy = accuracy/len(test_set)*100
        print(f"Test accuracy score: {total_accuracy}")
        
        # Save the training log and the extended weights for later inferences
        with open(f"{config_name}_acc:{total_accuracy}.network", "wb") as f:
            pickle.dump(network, f, pickle.HIGHEST_PROTOCOL)
        log = np.array([[tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor for tensor in sublist] for sublist in train_log])
        np.save(f"{config_name}_acc:{total_accuracy}.npy",log.T)