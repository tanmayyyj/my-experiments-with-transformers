import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import os 
from tqdm import tqdm 
import matplotlib.pyplot as plt 
import time

# hyperparameters
batch_size = 32 
block_size = 256 
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 512 # Embedding dimension
n_head = 6 # Number of attention heads
n_layer = 6 # Number of transformer blocks
dropout = 0.2
checkout_interval = 1000 
# ------------

encoding = tiktoken.get_encoding("gpt2")


vocab_size = encoding.n_vocab


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # tril is a buffer, meaning it's part of the module's state but not a parameter
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape # C is n_embd
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        
        # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = q @ k.transpose(-2,-1) * (k.shape[-1]**-0.5) 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # apply causality mask
        wei = F.softmax(wei, dim=-1) # (B, T, T) normalize to probabilities
        wei = self.dropout(wei) # apply dropout

        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Projection layer to combine outputs from multiple heads
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate outputs from all heads along the last dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out)) # Apply projection and dropout
        return out

class FeedFoward(nn.Module):
    """ linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # Expansion layer
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # Projection back to n_embd
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication (self-attention) followed by computation (feed-forward) """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) # LayerNorm before self-attention
        self.ln2 = nn.LayerNorm(n_embd) # LayerNorm before feed-forward

    def forward(self, x):
        # Apply residual connections
        x = x + self.sa(self.ln1(x)) # LayerNorm -> MultiHeadAttention -> Residual connection
        x = x + self.ffwd(self.ln2(x)) # LayerNorm -> FeedForward -> Residual connection
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # Token embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Positional embeddings
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Sequence of transformer blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # Final layer norm
        # Linear layer to project embeddings to vocabulary size (logits)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,n_embd)
        x = tok_emb + pos_emb # (B,T,n_embd) combines token and position information
        x = self.blocks(x) # (B,T,n_embd) pass through transformer blocks
        x = self.ln_f(x) # (B,T,n_embd) final layer norm
        logits = self.lm_head(x) # (B,T,vocab_size) project to vocabulary size

        loss = None
        if targets is not None:
            # Reshape logits for F.cross_entropy: (N, C) where N=B*T and C=vocab_size
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(model, start_tokens, max_new_tokens):
        model.eval()
        idx = start_tokens
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -block_size:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :]  # last token logits
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, next_token), dim=1)

                # Decode the last token only
                token_id = next_token.item()
                token_str = encoding.decode([token_id])

                # Print token without newline, flush buffer for live effect
                print(token_str, end='', flush=True)
                time.sleep(0.05)  # adjust delay for typing speed

        print()
        return idx