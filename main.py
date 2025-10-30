#%%
import torch
from torch import nn
from torch import optim 
from torch.utils.data import Dataset, DataLoader
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import os
from collections import Counter
import LstmLLM
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = 120
batch_size = 64
embeddin_dim = 128
hidden_dim = 256
num_layers = 4
dropout_embd = 0.05
dropout_rnn = 0.05
n_epochs = 10
lr = 0.04
weight_decay=1e-6
momentum=0.9

n_train_tokens_max = 100000


def prepare_data(input_data, seq_len):
    n_tokens = (len(input_data)//seq_len) * seq_len
    data = input_data[: n_tokens]
    return data.view(-1, seq_len)

class MyCustomDataset(Dataset):
    def __init__(self, inpot_tokens_2D):
        self.inpot_tokens_2D = inpot_tokens_2D

    def __len__(self):
        return self.inpot_tokens_2D.shape[0]
    
    def __getitem__(self, idx):
        sample = self.inpot_tokens_2D[idx]
        return sample[:-1], sample[1:] # (input, target)


print(torch.__version__)
print(torchtext.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

"""
data = load_dataset("wikitext", "wikitext-2-v1")
"""
# load dataset offline
address = r"M:\Git\DeepKatalisor\1_projekt_NLP_WikiText2P\dataset\wikitext-2"
data = load_dataset("text", data_files={
    "train": os.path.join(address, "wiki.train.tokens"),
    "validation":  os.path.join(address, "wiki.valid.tokens"),
    "test":  os.path.join(address, "wiki.test.tokens")
})
train_data = data['train']['text']
valid_data = data['validation']['text']
test_data = data['test']['text']

# create vocab also lookup tabel
tokenizer = get_tokenizer("basic_english")
def yield_tokens(data_iter):
    for line in data_iter:
        yield(tokenizer(line.lower()))
        




# other method to get tokenized vec---------------------------------------
"""
train_tokens = []
for line in train_data:
    if len(tokenizer(line))>0: # because tokenizer ('' or ' ' or Enter) give us []
        train_tokens.append(vocab(tokenizer(line.lower()))) 

# train_tokens list of lists with different size but we need for
# 2D-Tensor equall lenght of them, that is why we make concat elements
train_tokens_flatten =[token for line in train_tokens for token in line]
"""
#-------------------------------------------------------------------------
# map call first parameter as function of an itterable obj for example a list
# train_tokens: ['=', 'valkyria', 'chronicles', 'iii', '=', 'senjō', ...]
train_tokens = [t for train_data_tokenized in map(tokenizer, train_data) if train_data_tokenized != [] for t in train_data_tokenized]

start = time.time()
vocab = build_vocab_from_iterator([tokenizer(line.lower()) for line in train_data], specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
print(time.time()-start) #2.14

""" 
# other method but a little slowlly
start = time.time()
vocab = build_vocab_from_iterator(yield_tokens(train_data), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
print(time.time()-start) # 2.3
"""



train_tokens_idx = vocab(train_tokens)
train_tokens_idx = torch.tensor(train_tokens_idx, device=device)         

train_dataset = MyCustomDataset(prepare_data(train_tokens_idx, seq_len+1))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
next(iter(train_dataloader))
#----------------------- small test (kavosh haye maaan) ----------------------------
vocab.get_stoi()
print(f"train_dataset:{len(train_data)}  ,  vocab:{len(vocab)}")  
# train_dataset:36718  ,  vocab:28782
print(vocab(["Chronicles", "returned", "follows"])) # [0, 435, 1694]
tokenized = tokenizer("Chronicles returned 2") # ['chronicles', 'returned', '2']


freqs = Counter()
for tokens in map(tokenizer, train_data):
    freqs.update(tokens)    # number of repeat each token in train_data
freqs.most_common() #[('the', 130768), (',', 102615), ('.', 83397), ('of', 57030), ('<unk>', 54625),...]

print(next(iter(train_dataset))[0].shape) # 120
len(train_dataset) # 121

train_iter = iter(train_dataset)
print(next(train_iter))

x, y = next(iter(train_dataset))
x.shape # [120]
y.shape # [120]
#-------------------------------------------------------------------------

############ Make Model & Train ######################

model_lstm = LstmLLM.LstmLLm(len(vocab), embeddin_dim, hidden_dim, num_layers,
                           device, dropout_embd, dropout_rnn).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_lstm.parameters(), lr= lr, weight_decay=weight_decay, momentum=momentum)


######### my test ####################
next(model_lstm.parameters()).device

x_batch, y_batch = next(iter(train_dataloader))
print(x_batch.shape) # [64, 120]
print(y_batch.shape) # [64, 120]
print(x_batch.device)
print(y_batch.device)

print(torch.cuda.memory_allocated() / 1024**2, "MB allocated")
print(torch.cuda.memory_reserved() / 1024**2, "MB reserved")

################## train 1 ###########################
for epoch in range(n_epochs):
    losses_val = []
    losses_train = []
    print(f"\nEpoch {epoch}/{n_epochs} — Device: {device}")
    for i, (x_batch, y_batch) in enumerate(train_dataloader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        model_lstm.train()
        y_hat, _ = model_lstm(x_batch) # [64, 120, 28782]      
        # y_hat: (n_batch, seq_len, vocab_size)->(n_batch*seq_len, vocab_size)<->.view (-1,  vocab_size)
        y_hat = y_hat.view(-1, len(vocab))
        # target: y_batch_shape: (n_batch, seq_len)->(n_batch*seq_len)->.flatten()<->.view(-1)
        y_batch = y_batch.view(-1)
        train_loss = loss_fn(y_hat, y_batch)

        # Backpropagation
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        losses_train.append(train_loss)

        print(f"Batch {i} , loss={train_loss.item()}")
        print("---------------------------------------------------------------")
    if device.type == "cuda":
        print(f"GPU memory used: {torch.cuda.memory_allocated() / 1024**2:.1f} MB") 



################## train with tokens ###################

n_token = 0
iter_batch = iter(train_dataloader)
while n_token < n_train_tokens_max:
    x_batch, y_batch = next(iter_batch)
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    model_lstm.train()
    y_hat, _ = model_lstm(x_batch) 
    
    # y_hat: 3D:[batch_size, seq_len, vocab_size] -> 2D:[batch_size*seq_len, vocab_size]
    y_hat = y_hat.view (-1, len(vocab))
    
    # 2D:[batch_size, seq_len] -> 1D:[batch_size*seq_len]
    y_batch = y_batch.flatten()       
    
    loss_train = loss_fn(y_hat, y_batch)
    
    # Backpropagation
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    n_token += x_batch.shape[0] * x_batch.shape[1]
    n_batch = n_token // (batch_size*seq_len)
    print(f"Batch {n_batch} ,  n_token={n_token}  ,  loss={loss_train.item()}")











