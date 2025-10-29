#%%
import torch
from torch.utils.data import Dataset
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = 128
batch_size = 64

def prepare_data(input_data):
    n_tokens = (len(input_data)//seq_len) * seq_len
    data = input_data[: n_tokens]
    return data.view(-1, seq_len)

class MyCustomDataset(Dataset):
    def __init__(self, inpot_tokens_2D):
        self.inpot_tokens_2D = inpot_tokens_2D

    def __len__(self):
        return self.inpot_tokens_2D.shape[-1]
    
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
        yield(tokenizer(line))
        
vocab = build_vocab_from_iterator(yield_tokens(train_data), specials=["<unk>"])  
vocab.set_default_index(vocab["<unk>"])

   
print(f"train_dataset:{len(train_data)}  ,  vocab:{len(vocab)}")   
len(vocab)       
print(vocab(["Chronicles", "returned", "follows"])) # [0, 435, 1694]
     
tokenized = tokenizer("Chronicles returned 2") # ['chronicles', 'returned', '2']
 
train_tokens = []
for line in train_data:
    if len(tokenizer(line))>0: # because tokenizer ('' or ' ') give us []
        train_tokens.append(vocab(tokenizer(line))) 

# train_tokens list of lists with different size but we need for
# 2D-Tensor equall lenght of them, that is why we make concat elements
train_tokens_flatten =[token for line in train_tokens for token in line]

train_tokens = torch.tensor(train_tokens_flatten, device=device)         

train_dataset = MyCustomDataset(prepare_data(train_tokens))

it = next(iter(train_dataset))


len(train_tokens)





























