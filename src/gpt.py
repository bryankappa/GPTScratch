# credit to https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
import torch
import torch.nn as nn
from torch.nn import functional as F

#Parameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#Preprocess / find the unique text characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("".join(chars))
print("Vocab size: ", vocab_size)


# Create a mapping from character to index and vice versa ( it is basically a tokenizer)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


data = torch.tensor(encode(text), dtype=torch.long) # convert the text to a tensor

# train test split
n = int(0.9*len(data))
train_data, val_data = data[:n], data[n:]

# create a dataloader
def get_batch(split, device=None):
    data = train_data if split == 'train' else val_data # use train_data for training and val_data for validation, and uses array for the data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]) # the one dimensional tensor which is then stacked
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # stack the y values
    x, y = x.to(device), y.to(device) # this essentially labels the data and showcases what the predictions are.
    return x,y


# lets print them out to test it out
# xb, yb = get_batch('train')
# print(xb.shape)
# print(yb.shape)
# print('----')