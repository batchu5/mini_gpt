"""
Mini-GPT: a small character-level GPT (decoder-only transformer) built from scratch in PyTorch.

It reads a text file, learns to predict the next character, then generates new text
in the same style.

Run on Colab (GPU):   !python mini_gpt.py
Use your own text:    !python mini_gpt.py --data my_text.txt
Quick CPU check:      python mini_gpt.py --smoke
"""
import argparse
import os
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


# ---------------------------------------------------------------------------
# 1. Data: character-level tokenizer
# ---------------------------------------------------------------------------
def load_text(path):
    if not os.path.exists(path):
        print(f"{path} not found, downloading Tiny Shakespeare...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class CharTokenizer:
    """Maps every unique character to an integer id and back."""

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


def get_batch(data, block_size, batch_size, device):
    """Random chunks of text. y is x shifted by one: the model predicts the next char."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# 2. Model: the transformer pieces
# ---------------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    """
    Each token builds a query (what am I looking for?), key (what do I contain?)
    and value (what do I share?). Attention weights = softmax(Q K^T / sqrt(d_k)).
    A causal mask stops tokens from looking at future tokens.
    """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)  # Q, K, V for all heads in one matmul
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        # Lower-triangular mask: position t can only attend to positions <= t
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape  # batch, time (sequence length), channels (embedding size)
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim) so every head attends independently
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scale by sqrt(d_k) so dot products don't grow huge and saturate softmax
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        out = att @ v  # weighted sum of values
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # concatenate heads
        return self.resid_drop(self.proj(out))


class FeedForward(nn.Module):
    """Per-token MLP: expand 4x, non-linearity, project back."""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: attention (tokens talk) + feed-forward (tokens think).
    Pre-LayerNorm and residual connections keep deep stacks trainable."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)  # what the character is
        self.pos_emb = nn.Embedding(block_size, n_embd)  # where it is (learned positions)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)  # scores for every possible next char

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x = self.blocks(x)
        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive sampling: predict one char, append it, repeat."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]  # never feed more than the context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # only the last position predicts the next char
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ---------------------------------------------------------------------------
# 3. Training
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model, train_data, val_data, args, device):
    """Average loss over several batches, so the printed numbers aren't noisy."""
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(data, args.block_size, args.batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="input.txt", help="text file to train on (downloads Shakespeare if missing)")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--block_size", type=int, default=128, help="context length in characters")
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--gen_chars", type=int, default=500)
    p.add_argument("--smoke", action="store_true", help="tiny settings for a quick CPU check")
    args, _ = p.parse_known_args()  # ignore extra args Jupyter/Colab passes (e.g. -f kernel.json)

    if args.smoke:
        args.steps, args.batch_size, args.block_size = 50, 8, 32
        args.n_embd, args.n_head, args.n_layer = 32, 2, 2
        args.eval_interval, args.eval_iters, args.gen_chars = 25, 5, 100

    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    text = load_text(args.data)
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))  # 90% train, 10% validation
    train_data, val_data = data[:n], data[n:]
    print(f"Characters: {len(text):,} | vocab size: {tok.vocab_size}")

    model = MiniGPT(tok.vocab_size, args.block_size, args.n_embd, args.n_head, args.n_layer, args.dropout).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for step in range(args.steps + 1):
        if step % args.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, args, device)
            print(f"step {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
        if step == args.steps:
            break

        x, y = get_batch(train_data, args.block_size, args.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # avoid exploding gradients
        optimizer.step()

    torch.save({"model": model.state_dict(), "chars": tok.chars, "args": vars(args)}, "mini_gpt.pt")
    print("Saved model to mini_gpt.pt")

    print("\n--- Generated text ---")
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    model.eval()
    print(tok.decode(model.generate(start, args.gen_chars, temperature=0.8, top_k=20)[0].tolist()))


if __name__ == "__main__":
    main()
