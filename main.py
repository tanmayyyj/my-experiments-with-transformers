import torch
import tiktoken
from model import *
import argparse

parser = argparse.ArgumentParser(description="Generate text using GPT model.")
parser.add_argument("--max_new_tokens", type=int, default=250, help="Number of new tokens to generate.")
args = parser.parse_args()

max_new_tokens = args.max_new_tokens

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load('512_final_gpt_model.pth', map_location=device, weights_only=False)
model.eval()

# Load tokenizer
encoding = tiktoken.get_encoding("gpt2")

start_token_id = encoding.encode(" ")[0]
context = torch.tensor([[start_token_id]], dtype=torch.long, device=device)

generated_tokens = GPTLanguageModel.generate(model, context, max_new_tokens=max_new_tokens)[0].tolist()
decoded_text = encoding.decode(generated_tokens)

with open("generated_story.txt", "w", encoding="utf-8") as f:
    f.write(decoded_text)

