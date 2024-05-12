import torch
import torch.nn as nn
from torch.nn import functional as F
import time

#Parameters
head = 16

# def B_self_attn(q, k, v):
#     B, T, C = q.shape
#     q = q / (C**0.5)
#     wei = torch.bmm(q, k.transpose(1, 2))
#     wei = F.softmax(wei, dim=-1)
#     out = torch.bmm(wei, v)
#     return out

def A_self_attn(q, k, v):
    B, T, C = q.shape
    head_dim = C // head
    q = q.reshape(B, T, head, head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
    k = k.reshape(B, T, head, head_dim).transpose(1, 2)
    v = v.reshape(B, T, head, head_dim).transpose(1, 2)
    wei =  torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5) # q @ k^T is (B, num_heads, T, head_dim) @ (B, num_heads, head_dim, T) = (B, num_heads, T, T)
    wei = F.softmax(wei, dim=-1)
    out = torch.matmul(wei, v)
    out = out.transpose(1, 2).reshape(B, T, C)
    return out

# quick benchmark
def run_benchmark(iterations=1000):
    times = []
    memory_usage = []

    for _ in range(iterations):
        start_time = time.time()
        # Execute the code
        B_self_attn(torch.randn(4, 8, 16), torch.randn(4, 8, 16), torch.randn(4, 8, 16))
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
        
        torch.cuda.synchronize()  # For GPU
        memory_used = torch.cuda.max_memory_allocated() / 1024**2  # Also for GPU
        memory_usage.append(memory_used)
        torch.cuda.reset_peak_memory_stats()

    avg_time = sum(times) / len(times)
    avg_memory = sum(memory_usage) / len(memory_usage)
    print(f"Average execution time: {avg_time:.6f} seconds")
    print(f"Average peak memory usage: {avg_memory:.2f} MB")


run_benchmark(iterations=10)