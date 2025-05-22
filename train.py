import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import os # For torch.set_num_threads
from tqdm import tqdm # For the progress bar
import matplotlib.pyplot as plt # For plotting
from model import *

# torch.manual_seed(1337)

if __name__ == "__main__":
    model = GPTLanguageModel()
    m = model.to(device)
    print(f"{sum(p.numel() for p in m.parameters())/1e6:.2f} M parameters")

    if device == 'cpu':
        torch.set_num_threads(os.cpu_count())
        print(f"PyTorch using {torch.get_num_threads()} CPU threads.")

    with open('TinyStories-valid.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    encoding = tiktoken.get_encoding("gpt2")

    tokens = encoding.encode(text, allowed_special={'<|endoftext|>'})
    data = torch.tensor(tokens, dtype=torch.long)

    vocab_size = encoding.n_vocab 

    # Train and test splits
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # data loading function
    def get_batch(split):
        data_source = train_data if split == 'train' else val_data
        ix = torch.randint(len(data_source) - block_size, (batch_size,))
        x = torch.stack([data_source[i:i+block_size] for i in ix])
        y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval() # Set model to evaluation mode
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train() # Set model back to training mode
        return out

    # Create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_losses_log = []
    val_losses_log = []
    iters_log = []

    for iter in tqdm(range(max_iters)):  

        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            train_losses_log.append(losses['train'])
            val_losses_log.append(losses['val'])
            iters_log.append(iter)

        if iter > 0 and iter % checkout_interval == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iter': iter,
                'train_losses_log': train_losses_log,
                'val_losses_log': val_losses_log,
                'iters_log': iters_log,
                'train_loss': losses['train'],
                'val_loss': losses['val']
            }
            torch.save(checkpoint, f'checkpoint_iter_{iter}.pth')
            print(f"Checkpoint saved at iteration {iter}")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    final_model_path = 'final_gpt_model.pth'
    torch.save(model, final_model_path)
    print(f"\nFinal model weights saved to {final_model_path}")

    if train_losses_log and val_losses_log and iters_log:
        plt.figure(figsize=(10, 6))
        plt.plot(iters_log, train_losses_log, label='Training Loss')
        plt.plot(iters_log, val_losses_log, label='Validation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig('loss_plot.png')
        plt.show()
        print("\nLoss plot generated and saved as loss_plot.png.")
    else:
        print("\nNo loss data logged to plot. Ensure eval_interval is met at least once.")
