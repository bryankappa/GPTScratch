import torch
import torch.nn as nn
from torch.nn import functional as F

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

