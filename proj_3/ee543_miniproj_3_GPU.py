import numpy as np
import os
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import pickle
import itertools
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay

class UCI_HAR_Dataset():
    def __init__(self, dataset_path:str, val_sample_from_each_class:int=120):
        # Read and parse train data
        with open(os.path.join(dataset_path, 'train', 'X_train.txt')) as f:
            self.train_x = f.readlines()
        with open(os.path.join(dataset_path, 'train', 'y_train.txt')) as f:
            self.train_y = f.readlines()

        self.train_x = np.array([[float(i) for i in self.train_x[k].split(' ') if i!=''] for k in range(len(self.train_x))])
        self.train_y = [[int(i)-1 for i in self.train_y[k].split(' ') if i!=''] for k in range(len(self.train_y))]
        self.train_y = np.array([i for j in self.train_y for i in j])

        # Extract validation indices 
        sorted_train_y_indices = [np.where(self.train_y == element)[0] for element in range(6)]
        val_indices = np.concatenate([np.random.choice(sorted_train_y_indices[i], size=val_sample_from_each_class, replace=False) for i in range(6)]).astype(np.uint32)
        self.val_x = self.train_x[val_indices]
        self.val_y = self.train_y[val_indices]

        # Subtract the validation from train set
        mask = ~np.isin(np.arange(len(self.train_x)), val_indices)
        self.train_x = self.train_x[mask]
        self.train_y = self.train_y[mask]

        # Convert to one-hot encoded vectors
        self.val_y = np.eye(6)[self.val_y] 
        self.train_y = np.eye(6)[self.train_y]

        # Sanity check
        assert len(self.train_x) == 7352-val_sample_from_each_class*6 and len(self.train_y) == 7352-val_sample_from_each_class*6
        for i in range(len(self.train_x)):
            assert len(self.train_x[i])==561
        
        # To GPU
        self.train_x = torch.tensor(self.train_x, device='cuda:0').type(torch.float32)
        self.train_y = torch.tensor(self.train_y, device='cuda:0').type(torch.uint8)
        self.val_x = torch.tensor(self.val_x, device='cuda:0').type(torch.float32)
        self.val_y = torch.tensor(self.val_y, device='cuda:0').type(torch.uint8)

        # Read and parse test data
        with open(os.path.join(dataset_path, 'test', 'X_test.txt')) as f:
            self.test_x = f.readlines()
        with open(os.path.join(dataset_path, 'test', 'y_test.txt')) as f:
            self.test_y = f.readlines()

        self.test_x = [[float(i) for i in self.test_x[k].split(' ') if i!=''] for k in range(len(self.test_x))]
        self.test_y = [[int(i)-1 for i in self.test_y[k].split(' ') if i!=''] for k in range(len(self.test_y))]
        self.test_y = [i for j in self.test_y for i in j]
        # Convert to one-hot encoded vectors
        self.test_y = np.eye(6)[self.test_y]

        # Sanity check
        assert len(self.test_x) == 2947 and len(self.test_y) == 2947
        for i in range(len(self.test_x)):
            assert len(self.test_x[i])==561

        # To GPU
        self.test_x = torch.tensor(self.test_x, device='cuda:0').type(torch.float32)
        self.test_y = torch.tensor(self.test_y, device='cuda:0').type(torch.uint8)

    def shuffle_train(self):
        indices = torch.randperm(len(self.train_x))
        self.train_x = self.train_x[indices]
        self.train_y = self.train_y[indices]

class Dataloader:
    """
    Wrapper for the UCI_HAR_Dataset, to avoid re-loading the txt dataset files for train-val-test
    """
    def __init__(self, ds:UCI_HAR_Dataset, ds_type:str):
        self.ds_type = ds_type
        self.load_ds(ds, self.ds_type)

    def __getitem__(self, idx):
        if self.ds_type == 'train':
            return self.train_x[idx], self.train_y[idx]
        elif self.ds_type == 'val':
            return self.val_x[idx], self.val_y[idx]
        elif self.ds_type == 'test':
            return self.test_x[idx], self.test_y[idx]

    def load_ds(self, ds:UCI_HAR_Dataset, ds_type:str):
        if ds_type == 'train':
            self.train_x = ds.train_x
            self.train_y = ds.train_y
        if ds_type == 'val':
            self.val_x = ds.val_x
            self.val_y = ds.val_y
        if ds_type == 'test':
            self.test_x = ds.test_x
            self.test_y = ds.test_y

    def __len__(self):
        if self.ds_type == 'train':
            return len(self.train_x)
        if self.ds_type == 'val':
            return len(self.val_x)
        if self.ds_type == 'test':
            return len(self.test_x)

class LinearLayer():
    def __init__(self, in_feat, out_feat, act_type, lr, m_alpha, drop_rate):
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.act_type = act_type

        self._w_ext = 0.02 * torch.rand(size=(out_feat, in_feat+1), requires_grad=False, device='cuda:0') - 0.01
        self._w_ext[:,-1] = 0
        self._delta_w_prev = 0

        self._grad = torch.zeros((out_feat,1), requires_grad=False, device='cuda:0')
        self._gamma_prime = torch.eye(out_feat, out_feat, requires_grad=False, device='cuda:0')
        self._delta_w_list = []
        self._input = None
        self._minus_one = torch.ones(1, device='cuda:0', requires_grad=False)*-1
            
        self._eta = lr # Learning rate
        self.m_alpha = m_alpha # Momentum alpha
        self.dropout_rate = drop_rate

    def _act(self, v):
        if self.act_type == 'relu':
            return torch.max(torch.tensor(0.0, device='cuda:0', requires_grad=False), v)
        elif self.act_type == 'sigmoid':
            return 1/(1+torch.exp(-v))
        elif self.act_type == 'tanh':
            return torch.tanh(v)
            # NOTE Using exponentials yield very large numbers without any clip operations, resulting in NaN's. NaN/NaN is bad
            #a = torch.exp(v)
            #b = torch.exp(-v)
            #return (a-b)/(a+b)
        elif self.act_type == 'softmax':
            e = torch.exp(v - torch.max(v, dim=0, keepdim=True)[0])
            return e / e.sum(dim=0, keepdim=True)
    
    def _act_prime(self, o): # Closed-form of the activation derivatives
        if self.act_type == 'relu':  
            return torch.where(o>=0, torch.tensor(1.0, device='cuda:0', requires_grad=False), torch.tensor(0.0, device='cuda:0', requires_grad=False))
        elif self.act_type == 'sigmoid':
            return o*(1-o)
        elif self.act_type == 'tanh':
            return (1-o**2)
        elif self.act_type == 'softmax':
            return torch.diag(o) + -torch.outer(o, o) + torch.eye(o.shape[0], device='cuda:0')
    
    def apply_dropout(self, x):
        if self.dropout_rate > 0.0:
            self.mask = (torch.rand(x.size(), device='cuda:0') < self.dropout_rate).type(torch.float32)
            x = x * self.mask
        return x
    
    def forward(self, x, is_training=True):
        """
        Returns the matrix multiplication of extended weight matrix (_w_ext) with the extended input (_input) as the output.
        o(out_feat, 1) = _w_ext(out_feat x in_feat+1) @ _input(in_feat+1 x 1)
        Meanwhile, records the derivative of output activations (_gamma_prime(out_feat x out_feat) = diag(_act_prime(out_feat, 1))).
        """
        self._input = torch.cat([x,self._minus_one])
        if self.dropout_rate > 0.0 and is_training:
            v = self._w_ext @ self.apply_dropout(self._input)
        else:
            v = self._w_ext @ self._input
        o = self._act(v)
        
        if not is_training and self.dropout_rate > 0.0: # Activation scale for validation & test
            o = o * self.dropout_rate

        if self.act_type == 'relu':
            self._gamma_prime = torch.diag(self._act_prime(v))
        elif self.act_type == 'softmax':
            self._gamma_prime = self._act_prime(o)
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
        self._delta_w_list.append(delta_w)

    def step(self):
        """
        Updates the extended weight matrix by stepping towards the gradient descent direction.
        In case of batch learning, the step is the accumulation of the recorded steps in _delta_w_list.
        """
        delta_w = (sum(self._delta_w_list))/len(self._delta_w_list) + self.m_alpha*self._delta_w_prev
        self._w_ext = self._w_ext + delta_w
        self._delta_w_prev = delta_w
        self._delta_w_list = []

class Net():
    """
    A basic wrapper class for the neural network.
    """
    def __init__(self, is_training:bool, layer_nums:list, acts:list, learning_rate:float, momentum_alpha:float, drop_rate:float):
        self.is_training = is_training
        self.layer_nums = layer_nums
        self.out_num = layer_nums[-1][-1]
        self.perceptron = []
        self.eye = torch.eye(self.out_num, self.out_num, device='cuda:0', requires_grad=False)

        assert len(layer_nums) == len(acts)
        for i in range(len(layer_nums)): # Construct the perceptron layers
            if i==1: # Only apply dropout to the 2nd layer (among i=0,1,2)
                self.perceptron.append(LinearLayer(in_feat=layer_nums[i][0], out_feat=layer_nums[i][1], act_type=acts[i], 
                                                lr=learning_rate, m_alpha=momentum_alpha, drop_rate=drop_rate))
            else:
                self.perceptron.append(LinearLayer(in_feat=layer_nums[i][0], out_feat=layer_nums[i][1], act_type=acts[i], 
                                                lr=learning_rate, m_alpha=momentum_alpha, drop_rate=0.0))

    def forward(self, x):
        x_ = x
        for i in range(len(self.perceptron)):
            x_ = self.perceptron[i].forward(x_, self.is_training)
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


def extract_confusion(filename):
    ds = UCI_HAR_Dataset(dataset_path='UCI HAR Dataset/UCI HAR Dataset')
    test_set = Dataloader(ds=ds, ds_type='test')

    with open(filename, 'rb') as f:
        network = pickle.load(f)

    network.is_training = False
    topk_acc = [0]*6
    all_outputs = []
    all_labels = []
    test_pbar = tqdm(test_set, desc='Test session:')
    for idx, (x, d) in enumerate(test_pbar):
        all_labels.append(d.cpu().numpy())
        o = network.forward(x)

        max_index = torch.argmax(o)
        one_hot_out = torch.zeros_like(o)
        one_hot_out[max_index] = 1
        all_outputs.append(one_hot_out.cpu().numpy())

        sorted_out_idx = torch.argsort(o, descending=True)
        top_k = torch.argmax(d[sorted_out_idx])
        for i in range(top_k, 6):
            topk_acc[i] += 1
    topk_acc = [x/len(test_set) for x in topk_acc]
    print(f"Test top-k accuracy scores: {topk_acc}")
    
    cm = confusion_matrix(np.array(all_labels).argmax(axis=1),np.array(all_outputs).argmax(axis=1))
    disp = ConfusionMatrixDisplay(cm, display_labels=['Walk','Up','Down','Sit','Stand','Lay'])
    disp.plot(cmap='OrRd', values_format='.0f')
    BS, _, N2, LR, MOM, DROP = filename.split('/')[1].split('_')
    plt.title(f'BS:{BS.split(":")[-1]}, N2:{N2.split(":")[-1]}, LR:{LR.split(":")[-1]}, MOM:{MOM.split(":")[-1]}, DROP:{DROP.split(":")[-1]}')
    plt.savefig(f'VAL_CONFUSION_{filename.split("/")[1]}.png')

def train():
    # Hyperparameter combinations
    early_stopping = True
    epoch_num = 100
    lr_list = [0.01, 0.001] # learning rate 
    N1 = 300
    N2_list = [100, 200]
    momentum_list = [0.0, 0.09]
    bs_list = [1, 50] # batch_size
    drop_rate_list = [0.0, 0.5]
    for comb in itertools.product(*[lr_list,N2_list,momentum_list,bs_list,drop_rate_list]):
        lr = comb[0]
        N2 = comb[1]
        momentum_alpha = comb[2]
        batch_size = comb[3]
        drop_rate = comb[4]
        
        config_name = f"bs:{batch_size}_ep:{epoch_num}_hidden:{N2}_lr:{lr}_mom:{momentum_alpha}_drop:{drop_rate}"
        print(f'#### Config name: {config_name}')
        os.makedirs(config_name, exist_ok=True)

        ds = UCI_HAR_Dataset(dataset_path='UCI HAR Dataset/UCI HAR Dataset')
        train_set = Dataloader(ds=ds, ds_type='train')
        test_set = Dataloader(ds=ds, ds_type='test')
        val_set = Dataloader(ds=ds, ds_type='val')

        network = Net(layer_nums=[[561,N1],[N1,N2],[N2,6]], 
                      acts=['relu','relu','softmax'], 
                      learning_rate=lr,
                      momentum_alpha=momentum_alpha,
                      drop_rate=drop_rate,
                      is_training=True)

        pbar = tqdm(train_set, desc='Training session:')
        train_log = []
        val_log = []
        prev_train_E_avg = 0
        for curr_epoch in range(epoch_num):
            # Start training session
            network.is_training = True
            E_list = []
            for idx, (x, d) in enumerate(pbar):
                o = network.forward(x)
                e = d - o
                network.backward(e)
                if (idx+1) % batch_size == 0: # accumulate grads of size=batch_size, then update weights
                    network.step()
                E = -torch.log(o[d.argmax()]).mean() 
                E_list.append(E)              
                if idx % 250 == 0:
                    pbar.set_description(f"Training session. Epoch:{curr_epoch} E:{sum(E_list)/len(E_list)}")         
                if idx % 500 == 0:
                    train_log.append([idx+curr_epoch*len(train_set), E])
            train_E_avg = sum(E_list)/len(E_list)
            ds.shuffle_train()
            train_set.load_ds(ds, ds_type='train')
            pbar = tqdm(train_set, desc='Training session:')
            
            # Start val session
            network.is_training = False
            val_E_list = []
            val_topk_acc = [0]*6
            val_pbar = tqdm(val_set, desc='Val session:')
            for idx, (x, d) in enumerate(val_pbar):
                o = network.forward(x)
                E = -torch.log(o[d.argmax()]).mean() 
                val_E_list.append(E)
                sorted_out_idx = torch.argsort(o, descending=True)
                top_k = torch.argmax(d[sorted_out_idx])
                for i in range(top_k, 6):
                    val_topk_acc[i] += 1
            val_E_avg = sum(val_E_list)/len(val_E_list)
            val_topk_acc = [x/len(val_set) for x in val_topk_acc]
            val_log.append([idx+curr_epoch*len(val_set), val_E_avg])
            print(f"Val top-k accuracy scores: {val_topk_acc}, avg error: {val_E_avg}")

            diff = torch.abs(train_E_avg-val_E_avg)
            print(f"Val-train error difference: {diff}")

            diff_E = torch.abs(prev_train_E_avg-train_E_avg)
            print(f"Train error difference: {diff_E}")

            prev_train_E_avg = train_E_avg
            
            if early_stopping and curr_epoch > 50 and diff >= 0.05:
                break # early stoping

            # Save ckpt 
            with open(os.path.join(config_name, f"{curr_epoch}_{config_name}_valacc:{val_topk_acc}.network"), "wb") as f:
                pickle.dump(network, f, pickle.HIGHEST_PROTOCOL)

        # Start testing session
        topk_acc = [0]*6
        test_pbar = tqdm(test_set, desc='Test session:')
        for idx, (x, d) in enumerate(test_pbar):
            o = network.forward(x)
            sorted_out_idx = torch.argsort(o, descending=True)
            top_k = torch.argmax(d[sorted_out_idx])
            for i in range(top_k, 6):
                topk_acc[i] += 1
        topk_acc = [x/len(test_set) for x in topk_acc]
        print(f"Test top-k accuracy scores: {topk_acc}")
    
        # Save the training log and the extended weights for later inferences
        with open(os.path.join(config_name, f"{config_name}_acc:{topk_acc}.network"), "wb") as f:
            pickle.dump(network, f, pickle.HIGHEST_PROTOCOL)
        t_log = np.array([[tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor for tensor in sublist] for sublist in train_log])
        v_log = np.array([[tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor for tensor in sublist] for sublist in val_log])
        np.save(os.path.join(config_name, f"{config_name}_testacc:{topk_acc}.npy"),t_log.T)
        np.save(os.path.join(config_name, f"{config_name}_valacc:{val_topk_acc}.npy"),v_log.T)

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

    # Note that since seeds are fixed, the extraction of the validation set will be constant among different runs, and not be random anymore! Good for consistency
 
    # Please check the dataset path in:
    # ds = UCI_HAR_Dataset(dataset_path='UCI HAR Dataset/UCI HAR Dataset')
    train()