import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32
# ------------
# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embed) # Can optimize this to use embedding dimension for compression
        self.head = nn.Linear(n_embed, vocab_size)
    def forward(self, idx, targets=None):
        tok_emb = self.token_emb(idx) # (batch_size, block_size, vocab_size)
        logits = self.head(tok_emb)
        
        # you cant return cross entropy due to mismatch of tensors
        # RuntimeError: Expected target size [32, 65], got [32, 8]
        

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, -1)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss


class Trainer:
    def __init__(self, model, device, learning_rate):
        self.model = model.to(device) # Send the model to the device
        self.device = device # Store the device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # Create an optimizer

    def train(self, data, max_iters, eval_interval): # Train the model
        self.model.train() # Set the model to training mode
        for iter in range(max_iters): # Iterate over the dataset
            xb, yb = get_batch('train') # Get a batch of data
            xb, yb = xb.to(device), yb.to(device) # Send the data to the device
            logits, loss = self.model(xb, yb) # Forward pass
            self.optimizer.zero_grad() # Zero out the gradients
            loss.backward() # Backward pass
            self.optimizer.step() # Update the weights

            if iter % eval_interval == 0: # Evaluate the model
                print(f"iter {iter}, train loss: {loss.item()}") # Print the loss

    def generate(self, start_token, max_new_tokens):
        self.model.eval()  # Set the model to evaluation mode
        generated = [start_token]
        current_input = torch.tensor([start_token], dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _ = self.model(current_input)
                probs = F.softmax(logits[:, -1, :], dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated.append(next_token.item())
                # print(next_token.shape, current_input.shape)
                current_input = torch.cat((current_input, next_token), dim=1)
        return ''.join(itos[i] for i in generated)


if __name__ == '__main__':

    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]


    model = BigramLanguageModel(vocab_size)
    trainer = Trainer(model, device, learning_rate)

    trainer.train('train', max_iters, eval_interval)

    first_char_index = next(iter(stoi.values()))  # gets the first value from stoi
    generated_text = trainer.generate(first_char_index, max_new_tokens=500)
    print(generated_text)



